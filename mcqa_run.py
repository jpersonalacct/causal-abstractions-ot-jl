from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

from huggingface_hub import login as hf_login

from mcqa_experiment.compare_runner import CompareExperimentConfig, run_comparison
from mcqa_experiment.data import build_pair_banks, load_filtered_mcqa_pipeline
from mcqa_experiment.runtime import resolve_device


DEVICE = "cuda"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_mcqa"
OUTPUT_PATH = RUN_DIR / "mcqa_run_results.json"
SUMMARY_PATH = RUN_DIR / "mcqa_run_summary.txt"
SPLIT_PRINT_ORDER = ("train", "calibration", "test")

MODEL_NAME = "google/gemma-2-2b"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PROMPT_HF_LOGIN = True

# Data
MCQA_DATASET_PATH = "jchang153/copycolors_mcqa"
MCQA_DATASET_CONFIG = None
DATASET_SIZE = None  # Cap raw rows loaded from the dataset before factual filtering.
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # Applied after filtering to the retained pooled rows.
SPLIT_SEED = 0

# Experiment
METHODS = ["das"] #, "ot"]
TARGET_VARS = ["answer_pointer"] #, "answer"]
COUNTERFACTUAL_NAMES = ["answerPosition", "randomLetter", "answerPosition_randomLetter"]

LAYERS = "auto"
TOKEN_POSITION_IDS = ["last_token"] # "correct_symbol", "correct_symbol_period", 

BATCH_SIZE = 32 

RESOLUTION = 64 # gemma-2-2b has 2304 hidden layer size
OT_EPSILONS = [1.0]
UOT_BETA_ABSTRACTS = [0.1, 1.0]
UOT_BETA_NEURALS = [0.1, 1.0]
SIGNATURE_MODES = ["whole_vocab_kl_t1"] #, "answer_logit_delta"]
OT_TOP_K_VALUES = list(range(1, 11))
OT_LAMBDAS = [round(value * 0.1, 1) for value in range(1, 31)]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-3
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = [576, 1152, 1728, 2304]


def ensure_hf_login(token: str | None, prompt_login: bool) -> str | None:
    if token:
        hf_login(token=token, add_to_git_credential=False)
        return token
    if prompt_login:
        hf_login(add_to_git_credential=False)
    return token


def main() -> None:
    device = resolve_device(DEVICE)
    hf_token = ensure_hf_login(HF_TOKEN, PROMPT_HF_LOGIN)
    print(f"[run] starting MCQA run model={MODEL_NAME} device={device}")
    model, tokenizer, causal_model, token_positions, filtered_datasets = load_filtered_mcqa_pipeline(
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        dataset_size=DATASET_SIZE,
        hf_token=hf_token,
        dataset_path=MCQA_DATASET_PATH,
        dataset_name=MCQA_DATASET_CONFIG,
    )
    print("[run] building pair banks")
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=filtered_datasets,
        counterfactual_names=tuple(COUNTERFACTUAL_NAMES),
        target_vars=tuple(TARGET_VARS),
        split_ratios=SPLIT_RATIOS,
        split_seed=SPLIT_SEED,
    )
    print(f"[run] built splits={list(banks_by_split.keys())}")
    for split in SPLIT_PRINT_ORDER:
        split_metadata = data_metadata.get(split)
        if isinstance(split_metadata, dict) and split_metadata:
            sample_stats = next(iter(split_metadata.values()))
            print(
                f"{split} pair bank | total_pairs={int(sample_stats.get('size', 0))} "
                f"| datasets={sample_stats.get('dataset_names', [])}"
            )
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
    all_payloads = []
    for signature_mode in SIGNATURE_MODES:
        for epsilon in OT_EPSILONS:
            methods = tuple(method for method in METHODS if method != "uot")
            config = CompareExperimentConfig(
                model_name=MODEL_NAME,
                output_path=RUN_DIR / f"mcqa_sig-{signature_mode}_eps-{epsilon:g}.json",
                summary_path=RUN_DIR / f"mcqa_sig-{signature_mode}_eps-{epsilon:g}.txt",
                methods=methods,
                target_vars=tuple(TARGET_VARS),
                batch_size=BATCH_SIZE,
                ot_epsilon=float(epsilon),
                signature_mode=signature_mode,
                ot_top_k_values=tuple(OT_TOP_K_VALUES),
                ot_lambdas=tuple(OT_LAMBDAS),
                das_max_epochs=DAS_MAX_EPOCHS,
                das_min_epochs=DAS_MIN_EPOCHS,
                das_plateau_patience=DAS_PLATEAU_PATIENCE,
                das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
                das_learning_rate=DAS_LEARNING_RATE,
                das_subspace_dims=tuple(DAS_SUBSPACE_DIMS),
                resolution=RESOLUTION,
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
                )
            )
            if "uot" in METHODS:
                for beta_abstract in UOT_BETA_ABSTRACTS:
                    for beta_neural in UOT_BETA_NEURALS:
                        uot_config = CompareExperimentConfig(
                            model_name=MODEL_NAME,
                            output_path=RUN_DIR / (
                                f"mcqa_uot_sig-{signature_mode}_eps-{epsilon:g}_"
                                f"ba-{beta_abstract:g}_bn-{beta_neural:g}.json"
                            ),
                            summary_path=RUN_DIR / (
                                f"mcqa_uot_sig-{signature_mode}_eps-{epsilon:g}_"
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
                            resolution=RESOLUTION,
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
                            )
                        )
    from mcqa_experiment.runtime import write_json

    write_json(OUTPUT_PATH, {"runs": all_payloads})
    print(f"Wrote aggregate MCQA run payload to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
