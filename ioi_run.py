from __future__ import annotations

from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import os

from huggingface_hub import login as hf_login
import torch

from ioi_experiment.compare_runner import CompareExperimentConfig, run_comparison
from ioi_experiment.data import build_pair_banks, load_filtered_ioi_pipeline
from ioi_experiment.ot import (
    OTConfig,
    load_prepared_alignment_artifacts,
    prepare_alignment_artifacts,
    save_prepared_alignment_artifacts,
)
from ioi_experiment.runtime import resolve_device
from ioi_experiment.sites import enumerate_residual_sites


DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_ioi"
OUTPUT_PATH = RUN_DIR / "ioi_run_results.json"
SUMMARY_PATH = RUN_DIR / "ioi_run_summary.txt"
SIGNATURES_DIR = Path("signatures")
SPLIT_PRINT_ORDER = ("train", "calibration", "test")

MODEL_NAME = "google/gemma-2-2b"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PROMPT_HF_LOGIN = True

# Data
IOI_DATASET_PATH = "mib-bench/ioi"
DATASET_SIZE = 2000  # Cap raw rows loaded before factual filtering.
SPLIT_SEED = 0
TRAIN_POOL_SIZE = 200
CALIBRATION_POOL_SIZE = 100
TEST_POOL_SIZE = 100

# Experiment
METHODS = ["ot"]
TARGET_VARS = ["answer_index"]
# Names match the counterfactual keys in the MIB IOI dataset (without "_counterfactual" suffix).
COUNTERFACTUAL_NAMES = ["token", "position"]

# IOI token positions: s1 (first subject mention), io (indirect object), s2 (second subject), last_token
LAYERS = [25]
TOKEN_POSITION_IDS = ["s1", "io", "s2", "last_token"]

BATCH_SIZE = 64

RESOLUTIONS = [1]  # gemma-2-2b has hidden_size=2304
OT_EPSILONS = [20, 30, 40, 50]
UOT_BETA_ABSTRACTS = [0.1, 1.0]
UOT_BETA_NEURALS = [0.1, 1.0]
SIGNATURE_MODES = ["answer_logit_delta"]
OT_TOP_K_VALUES = list(range(1, 11))
OT_LAMBDAS = [i for i in range(1, 31)]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-3
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = [576, 1152, 1728, 2304]


def _signature_cache_spec(
    *,
    train_bank,
    resolution: int,
    signature_mode: str,
    selected_layers: list[int],
    token_position_ids: tuple[str, ...],
) -> dict[str, object]:
    train_rows_digest = hashlib.sha256(
        "\n".join(
            f"{base.get('raw_input', '')}|||{source.get('raw_input', '')}"
            for base, source in zip(train_bank.base_inputs, train_bank.source_inputs)
        ).encode("utf-8")
    ).hexdigest()
    return {
        "kind": "ioi_alignment_signatures",
        "model_name": MODEL_NAME,
        "dataset_path": IOI_DATASET_PATH,
        "counterfactual_names": list(COUNTERFACTUAL_NAMES),
        "split_seed": int(SPLIT_SEED),
        "resolution": int(resolution),
        "signature_mode": str(signature_mode),
        "train_pool_size": int(TRAIN_POOL_SIZE),
        "train_bank": train_bank.metadata(),
        "train_rows_digest": train_rows_digest,
        "selected_layers": [int(layer) for layer in selected_layers],
        "token_position_ids": list(token_position_ids),
        "batch_size": int(BATCH_SIZE),
    }


def _signature_cache_path(*, resolution: int, signature_mode: str, cache_spec: dict[str, object]) -> Path:
    spec_json = json.dumps(cache_spec, sort_keys=True, separators=(",", ":"))
    spec_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()[:12]
    stem = (
        f"ioi_res-{int(resolution)}_sig-{str(signature_mode)}"
        f"_train-{int(cache_spec['train_pool_size'])}_{spec_hash}.pt"
    )
    return SIGNATURES_DIR / stem


def ensure_hf_login(token: str | None, prompt_login: bool) -> str | None:
    if token:
        hf_login(token=token, add_to_git_credential=False)
        return token
    if prompt_login:
        hf_login(add_to_git_credential=False)
    return token


def _is_memory_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.OutOfMemoryError) or any(
        needle in message
        for needle in (
            "out of memory",
            "cuda out of memory",
            "mps backend out of memory",
            "cublas_status_alloc_failed",
            "cuda error: out of memory",
            "hip out of memory",
        )
    )


def _clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main() -> None:
    device = resolve_device(DEVICE)
    hf_token = ensure_hf_login(HF_TOKEN, PROMPT_HF_LOGIN)
    print(f"[run] starting IOI run model={MODEL_NAME} device={device}")
    model, tokenizer, causal_model, token_positions, filtered_datasets = load_filtered_ioi_pipeline(
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        dataset_size=DATASET_SIZE,
        hf_token=hf_token,
        dataset_path=IOI_DATASET_PATH,
    )
    print("[run] building pair banks")
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=filtered_datasets,
        counterfactual_names=tuple(COUNTERFACTUAL_NAMES),
        target_vars=tuple(TARGET_VARS),
        split_seed=SPLIT_SEED,
        train_pool_size=TRAIN_POOL_SIZE,
        calibration_pool_size=CALIBRATION_POOL_SIZE,
        test_pool_size=TEST_POOL_SIZE,
    )
    print(f"[run] built splits={list(banks_by_split.keys())}")
    for split in SPLIT_PRINT_ORDER:
        split_metadata = data_metadata.get(split)
        if isinstance(split_metadata, dict) and split_metadata:
            dataset_names = next(iter(split_metadata.values())).get("dataset_names", [])
            print(f"{split} pair bank | datasets={dataset_names}")
            for variable, variable_stats in sorted(split_metadata.items()):
                total_pairs = int(variable_stats.get("size", 0))
                changed_count = int(variable_stats.get("changed_count", 0))
                unchanged_count = max(0, total_pairs - changed_count)
                print(
                    f"{split} pair bank [{variable}] | changed={changed_count} "
                    f"| unchanged={unchanged_count} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                )
    selected_layers = list(range(int(model.config.num_hidden_layers))) if LAYERS == "auto" else list(LAYERS)
    print(f"[run] selected_layers={selected_layers}")
    token_position_ids = tuple(token_position.id for token_position in token_positions)
    all_payloads = []
    for method in METHODS:
        for signature_mode in SIGNATURE_MODES:
            for resolution in RESOLUTIONS:
                prepared_ot_artifacts = None
                try:
                    if method in {"ot", "uot"} and TARGET_VARS:
                        ot_sites = enumerate_residual_sites(
                            num_layers=int(model.config.num_hidden_layers),
                            hidden_size=int(model.config.hidden_size),
                            token_position_ids=token_position_ids,
                            resolution=int(resolution),
                            layers=tuple(selected_layers),
                            selected_token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                        )
                        train_bank = banks_by_split["train"][TARGET_VARS[0]]
                        cache_spec = _signature_cache_spec(
                            train_bank=train_bank,
                            resolution=int(resolution),
                            signature_mode=signature_mode,
                            selected_layers=selected_layers,
                            token_position_ids=token_position_ids,
                        )
                        cache_path = _signature_cache_path(
                            resolution=int(resolution),
                            signature_mode=signature_mode,
                            cache_spec=cache_spec,
                        )
                        prepared_ot_artifacts = load_prepared_alignment_artifacts(
                            cache_path,
                            expected_spec=cache_spec,
                        )
                        if prepared_ot_artifacts is not None:
                            print(
                                f"[signatures] loaded cache path={cache_path} "
                                f"prepare_time={float(prepared_ot_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                            )
                        else:
                            prepared_ot_artifacts = prepare_alignment_artifacts(
                                model=model,
                                fit_bank=train_bank,
                                sites=ot_sites,
                                device=device,
                                config=OTConfig(
                                    method=method,
                                    batch_size=BATCH_SIZE,
                                    epsilon=1.0,
                                    signature_mode=signature_mode,
                                    top_k_values=tuple(OT_TOP_K_VALUES),
                                    lambda_values=tuple(OT_LAMBDAS),
                                ),
                            )
                            prepared_ot_artifacts["cache_spec"] = cache_spec
                            prepared_ot_artifacts["cache_path"] = str(cache_path)
                            save_prepared_alignment_artifacts(
                                cache_path,
                                prepared_artifacts=prepared_ot_artifacts,
                                cache_spec=cache_spec,
                            )
                            print(
                                f"[signatures] saved cache path={cache_path} "
                                f"prepare_time={float(prepared_ot_artifacts.get('prepare_runtime_seconds', 0.0)):.2f}s"
                            )
                    epsilon_values = OT_EPSILONS if method in {"ot", "uot"} else [None]
                    for epsilon in epsilon_values:
                        if method == "uot":
                            for beta_abstract in UOT_BETA_ABSTRACTS:
                                for beta_neural in UOT_BETA_NEURALS:
                                    uot_config = CompareExperimentConfig(
                                        model_name=MODEL_NAME,
                                        output_path=RUN_DIR / (
                                            f"ioi_uot_res-{int(resolution)}_sig-{signature_mode}_eps-{float(epsilon):g}_"
                                            f"ba-{beta_abstract:g}_bn-{beta_neural:g}.json"
                                        ),
                                        summary_path=RUN_DIR / (
                                            f"ioi_uot_res-{int(resolution)}_sig-{signature_mode}_eps-{float(epsilon):g}_"
                                            f"ba-{beta_abstract:g}_bn-{beta_neural:g}.txt"
                                        ),
                                        methods=("uot",),
                                        target_vars=tuple(TARGET_VARS),
                                        batch_size=BATCH_SIZE,
                                        ot_epsilon=float(epsilon),
                                        uot_beta_abstract=float(beta_abstract),
                                        uot_beta_neural=float(beta_neural),
                                        signature_mode=signature_mode,
                                        ot_top_k_values=tuple(OT_TOP_K_VALUES),
                                        ot_lambdas=tuple(OT_LAMBDAS),
                                        resolution=int(resolution),
                                        layers=tuple(selected_layers),
                                        token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                                    )
                                    all_payloads.append(
                                        run_comparison(
                                            model=model,
                                            tokenizer=tokenizer,
                                            token_positions=token_positions,
                                            banks_by_split=banks_by_split,
                                            data_metadata=data_metadata,
                                            device=device,
                                            config=uot_config,
                                            prepared_ot_artifacts=prepared_ot_artifacts,
                                        )
                                    )
                        else:
                            output_stem = (
                                f"ioi_{method}_res-{int(resolution)}_sig-{signature_mode}"
                                if method != "ot"
                                else f"ioi_res-{int(resolution)}_sig-{signature_mode}"
                            )
                            if epsilon is not None:
                                output_stem = f"{output_stem}_eps-{float(epsilon):g}"
                            config = CompareExperimentConfig(
                                model_name=MODEL_NAME,
                                output_path=RUN_DIR / f"{output_stem}.json",
                                summary_path=RUN_DIR / f"{output_stem}.txt",
                                methods=(method,),
                                target_vars=tuple(TARGET_VARS),
                                batch_size=BATCH_SIZE,
                                ot_epsilon=float(epsilon) if epsilon is not None else 1.0,
                                signature_mode=signature_mode,
                                ot_top_k_values=tuple(OT_TOP_K_VALUES),
                                ot_lambdas=tuple(OT_LAMBDAS),
                                das_max_epochs=DAS_MAX_EPOCHS,
                                das_min_epochs=DAS_MIN_EPOCHS,
                                das_plateau_patience=DAS_PLATEAU_PATIENCE,
                                das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
                                das_learning_rate=DAS_LEARNING_RATE,
                                das_subspace_dims=tuple(DAS_SUBSPACE_DIMS),
                                resolution=int(resolution),
                                layers=tuple(selected_layers),
                                token_position_ids=None if TOKEN_POSITION_IDS is None else tuple(TOKEN_POSITION_IDS),
                            )
                            all_payloads.append(
                                run_comparison(
                                    model=model,
                                    tokenizer=tokenizer,
                                    token_positions=token_positions,
                                    banks_by_split=banks_by_split,
                                    data_metadata=data_metadata,
                                    device=device,
                                    config=config,
                                    prepared_ot_artifacts=prepared_ot_artifacts,
                                )
                            )
                except Exception as exc:
                    if not _is_memory_error(exc):
                        raise
                    print(
                        f"[oom] skipping method={method} signature_mode={signature_mode} resolution={int(resolution)} "
                        f"after memory failure: {exc}"
                    )
                    prepared_ot_artifacts = None
                    _clear_torch_memory()
                    continue
                finally:
                    prepared_ot_artifacts = None
                    _clear_torch_memory()
    from ioi_experiment.runtime import write_json

    write_json(OUTPUT_PATH, {"runs": all_payloads})
    print(f"Wrote aggregate IOI run payload to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
