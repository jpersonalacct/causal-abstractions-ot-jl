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

MODEL_NAME = "google/gemma-2-2b"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
PROMPT_HF_LOGIN = True
METHODS = ["das", "ot"]
TARGET_VARS = ["answer_pointer", "answer"]
COUNTERFACTUAL_NAMES = ["answerPosition", "randomLetter", "answerPosition_randomLetter"]

LAYERS = list(range(26))
TOKEN_POSITION_IDS = ["correct_symbol", "correct_symbol_period", "last_token"]

BATCH_SIZE = 8
DATASET_SIZE = None  # Use the full public HF train/validation/test splits before factual filtering.

RESOLUTION = 1 # gemma-2-2b has 2304 hidden layer size
OT_EPSILONS = [1.0]
OT_TAUS = [1.0]
UOT_BETA_ABSTRACTS = [0.1, 1.0]
UOT_BETA_NEURALS = [0.1, 1.0]
SIGNATURE_MODES = ["whole_vocab_kl_t1"] #, "answer_logit_delta"]
OT_TOP_K_VALUES = list(range(1, 11))
OT_LAMBDAS = [round(value * 0.1, 1) for value in range(1, 31)]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = [1, 4, 8, 16, 32]


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
    model, tokenizer, causal_model, token_positions, filtered_datasets = load_filtered_mcqa_pipeline(
        model_name=MODEL_NAME,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        dataset_size=DATASET_SIZE,
        hf_token=hf_token,
    )
    banks_by_split, data_metadata = build_pair_banks(
        tokenizer=tokenizer,
        causal_model=causal_model,
        token_positions=token_positions,
        datasets_by_name=filtered_datasets,
        counterfactual_names=tuple(COUNTERFACTUAL_NAMES),
        target_vars=tuple(TARGET_VARS),
    )
    all_payloads = []
    for signature_mode in SIGNATURE_MODES:
        for epsilon in OT_EPSILONS:
            for tau in OT_TAUS:
                methods = tuple(method for method in METHODS if method != "uot")
                config = CompareExperimentConfig(
                    model_name=MODEL_NAME,
                    output_path=RUN_DIR / f"mcqa_sig-{signature_mode}_eps-{epsilon:g}_tau-{tau:g}.json",
                    summary_path=RUN_DIR / f"mcqa_sig-{signature_mode}_eps-{epsilon:g}_tau-{tau:g}.txt",
                    methods=methods,
                    target_vars=tuple(TARGET_VARS),
                    batch_size=BATCH_SIZE,
                    ot_epsilon=float(epsilon),
                    ot_tau=float(tau),
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
                    layers=None if LAYERS is None else tuple(LAYERS),
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
                for beta_abstract in UOT_BETA_ABSTRACTS:
                    for beta_neural in UOT_BETA_NEURALS:
                        uot_config = CompareExperimentConfig(
                            model_name=MODEL_NAME,
                            output_path=RUN_DIR / (
                                f"mcqa_uot_sig-{signature_mode}_eps-{epsilon:g}_tau-{tau:g}_"
                                f"ba-{beta_abstract:g}_bn-{beta_neural:g}.json"
                            ),
                            summary_path=RUN_DIR / (
                                f"mcqa_uot_sig-{signature_mode}_eps-{epsilon:g}_tau-{tau:g}_"
                                f"ba-{beta_abstract:g}_bn-{beta_neural:g}.txt"
                            ),
                            methods=("uot",),
                            target_vars=tuple(TARGET_VARS),
                            batch_size=BATCH_SIZE,
                            ot_epsilon=float(epsilon),
                            ot_tau=float(tau),
                            uot_beta_abstract=float(beta_abstract),
                            uot_beta_neural=float(beta_neural),
                            signature_mode=signature_mode,
                            ot_top_k_values=tuple(OT_TOP_K_VALUES),
                            ot_lambdas=tuple(OT_LAMBDAS),
                            resolution=RESOLUTION,
                            layers=None if LAYERS is None else tuple(LAYERS),
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
