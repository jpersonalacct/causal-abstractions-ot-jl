import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

from addition_experiment.backbone import AdditionTrainConfig, load_backbone, train_backbone
from addition_experiment.ot_gradient import OTGradientConfig, run_alignment_gradient_pipeline
from addition_experiment.pair_bank import build_pair_bank
from addition_experiment.plots import save_comparison_plots
from addition_experiment.reporting import (
    build_method_selection_summary,
    format_method_candidate_sweep,
    format_method_selection_summary,
    print_results_table,
    summarize_method_records,
    write_text_report,
)
from addition_experiment.runtime import resolve_device, write_json
from addition_experiment.scm import load_addition_problem
from addition_experiment.seed_sweep import (
    build_seed_sweep_payload,
    format_seed_sweep_summary,
    save_seed_sweep_plots,
)


SEEDS = (42,)
DEVICE = "mps"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_addition_gradient"
CHECKPOINT_PATH_TEMPLATE = "models/addition_mlp_seed{seed}.pt"
OUTPUT_PATH = RUN_DIR / "addition_run_gradient_results.json"
SUMMARY_PATH = RUN_DIR / "addition_run_gradient_summary.txt"
RETRAIN_BACKBONES = False

# Methods included in the gradient-policy comparison run.
METHODS = ("ot",)  # Expand if you later add gradient variants for GW / FGW.

# Backbone training hyperparameters.
FACTUAL_TRAIN_SIZE = 30000
FACTUAL_VALIDATION_SIZE = 4000
HIDDEN_DIMS = (192, 192, 192, 192)
TARGET_VARS = ("S1", "C1", "S2", "C2")
LEARNING_RATE = 1e-3
EPOCHS = 200
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256

# Pair-bank split sizes.
TRAIN_PAIR_SIZE = 100
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 5000

# Training pair-bank construction hyperparameters.
# `TRAIN_PAIR_POLICY`:
#   - "mixed": target the positive carry fraction specified below
#   - "unfiltered": sample ordered pairs without balancing
# `*_PAIR_POLICY_TARGET` can be one of:
#   - "any", "C1", "C2", "both", "C1_only", "C2_only"
# `TRAIN_PAIR_POOL_SIZE`: number of unique digit rows sampled before constructing ordered pairs.
# Pair filtering is always based on carry interventions (`C1` / `C2`) only, even if
# `TARGET_VARS` includes additional variables such as `S1` or `S2`.
TRAIN_PAIR_POLICY_TARGET = "any"
TRAIN_MIXED_POSITIVE_FRACTION = 0.5
TRAIN_PAIR_POLICY = "unfiltered"
TRAIN_PAIR_POOL_SIZE = 256

# Calibration pair-bank construction hyperparameters.
CALIBRATION_PAIR_POLICY_TARGET = "any"
CALIBRATION_MIXED_POSITIVE_FRACTION = 0.5
CALIBRATION_PAIR_POLICY = "unfiltered"  # "mixed" # "unfiltered"
CALIBRATION_PAIR_POOL_SIZE = 256

# Test pair-bank construction hyperparameters.
TEST_PAIR_POLICY_TARGET = "any"
TEST_MIXED_POSITIVE_FRACTION = 0.5
TEST_PAIR_POLICY = "unfiltered"  # "mixed" # "unfiltered"
TEST_PAIR_POOL_SIZE = 256

# Shared evaluation batch size.
BATCH_SIZE = 128

# Shared alignment hyperparameters.
RESOLUTION = 1
FGW_ALPHA = 0.5

# Gradient policy-search hyperparameters.
POLICY_LEARNING_RATE = 5e-2      # Adam LR for the learned cutoff c and intervention strength lambda.
POLICY_EPOCHS = 1000             # Max calibration optimization epochs per (variable, layer) candidate.
POLICY_MIN_EPOCHS = 25           # Do not early-stop before this many epochs.
POLICY_PLATEAU_PATIENCE = 5      # Stop after this many calibration-eval steps without sufficient CE improvement.
POLICY_PLATEAU_REL_DELTA = 5e-3  # Relative CE improvement required to reset the plateau counter.
POLICY_TEMPERATURE = 0.5         # Softness of the ranked cutoff gate; lower values behave more like hard top-k.
POLICY_EVAL_INTERVAL = 1         # How often to score the current soft policy on the full calibration bank.
FIXED_TOP_K = 192                # Set to an int to keep top-k fixed within each layer; use None to learn it.
FIXED_LAMBDA = 1.0               # Set to a float to keep lambda fixed; use None to learn lambda.


@dataclass(frozen=True)
class GradientRunConfig:
    seed: int
    checkpoint_path: Path
    output_path: Path
    summary_path: Path
    methods: tuple[str, ...]
    factual_validation_size: int
    train_pair_size: int
    calibration_pair_size: int
    test_pair_size: int
    target_vars: tuple[str, ...]
    train_pair_policy: str
    train_pair_policy_target: str
    train_mixed_positive_fraction: float
    train_pair_pool_size: int | None
    calibration_pair_policy: str
    calibration_pair_policy_target: str
    calibration_mixed_positive_fraction: float
    calibration_pair_pool_size: int | None
    test_pair_policy: str
    test_pair_policy_target: str
    test_mixed_positive_fraction: float
    test_pair_pool_size: int | None
    batch_size: int
    resolution: int
    fgw_alpha: float
    policy_learning_rate: float
    policy_epochs: int
    policy_min_epochs: int
    policy_plateau_patience: int
    policy_plateau_rel_delta: float
    policy_temperature: float
    policy_eval_interval: int
    fixed_top_k: int | None
    fixed_lambda: float | None


def build_train_config(seed: int) -> AdditionTrainConfig:
    return AdditionTrainConfig(
        seed=seed,
        n_train=FACTUAL_TRAIN_SIZE,
        n_validation=FACTUAL_VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
    )


def build_run_config(seed: int, checkpoint_path: Path, run_dir: Path) -> GradientRunConfig:
    return GradientRunConfig(
        seed=seed,
        checkpoint_path=checkpoint_path,
        output_path=run_dir / "addition_run_gradient_results.json",
        summary_path=run_dir / "addition_run_gradient_summary.txt",
        methods=tuple(METHODS),
        factual_validation_size=FACTUAL_VALIDATION_SIZE,
        train_pair_size=TRAIN_PAIR_SIZE,
        calibration_pair_size=CALIBRATION_PAIR_SIZE,
        test_pair_size=TEST_PAIR_SIZE,
        target_vars=tuple(TARGET_VARS),
        train_pair_policy=TRAIN_PAIR_POLICY,
        train_pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        train_mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        train_pair_pool_size=TRAIN_PAIR_POOL_SIZE,
        calibration_pair_policy=CALIBRATION_PAIR_POLICY,
        calibration_pair_policy_target=CALIBRATION_PAIR_POLICY_TARGET,
        calibration_mixed_positive_fraction=CALIBRATION_MIXED_POSITIVE_FRACTION,
        calibration_pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
        test_pair_policy=TEST_PAIR_POLICY,
        test_pair_policy_target=TEST_PAIR_POLICY_TARGET,
        test_mixed_positive_fraction=TEST_MIXED_POSITIVE_FRACTION,
        test_pair_pool_size=TEST_PAIR_POOL_SIZE,
        batch_size=BATCH_SIZE,
        resolution=RESOLUTION,
        fgw_alpha=FGW_ALPHA,
        policy_learning_rate=POLICY_LEARNING_RATE,
        policy_epochs=POLICY_EPOCHS,
        policy_min_epochs=POLICY_MIN_EPOCHS,
        policy_plateau_patience=POLICY_PLATEAU_PATIENCE,
        policy_plateau_rel_delta=POLICY_PLATEAU_REL_DELTA,
        policy_temperature=POLICY_TEMPERATURE,
        policy_eval_interval=POLICY_EVAL_INTERVAL,
        fixed_top_k=FIXED_TOP_K,
        fixed_lambda=FIXED_LAMBDA,
    )


def load_or_train_backbone(problem, device, seed: int, checkpoint_path: Path):
    train_config = build_train_config(seed)
    if RETRAIN_BACKBONES or not checkpoint_path.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
        return model, backbone_meta, "trained"

    model, _, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=checkpoint_path,
        device=device,
        train_config=train_config,
    )
    return model, backbone_meta, "loaded"


def print_loaded_backbone_validation(backbone_meta: dict[str, object]) -> None:
    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    exact_acc = float(factual_metrics.get("exact_acc", 0.0))
    num_examples = int(factual_metrics.get("num_examples", 0))
    print(
        "Loaded backbone factual validation "
        f"| exact_acc={exact_acc:.4f} "
        f"| num_examples={num_examples}"
    )


def _build_gradient_summary_lines(
    config: GradientRunConfig,
    device,
    backbone_meta: dict[str, object],
    train_bank,
    calibration_bank,
    test_bank,
    method_payloads: dict[str, dict[str, object]],
    method_runtime_seconds: dict[str, float],
    summary_records: list[dict[str, object]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    method_selections = {
        method: build_method_selection_summary(method, method_payloads[method])
        for method in config.methods
    }
    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    summary_lines = [
        "Addition Gradient Compare Summary",
        f"checkpoint: {config.checkpoint_path}",
        f"seed: {config.seed}",
        f"device: {device}",
        f"target_vars: {', '.join(config.target_vars)}",
        (
            "pair_sizes: "
            f"train={config.train_pair_size}, "
            f"calibration={config.calibration_pair_size}, "
            f"test={config.test_pair_size}"
        ),
        (
            "train_pair_construction: "
            f"policy={config.train_pair_policy}, "
            f"target={config.train_pair_policy_target}, "
            f"mixed_positive_fraction={float(config.train_mixed_positive_fraction):.4f}, "
            f"pool_size={config.train_pair_pool_size}"
        ),
        (
            "calibration_pair_construction: "
            f"policy={config.calibration_pair_policy}, "
            f"target={config.calibration_pair_policy_target}, "
            f"mixed_positive_fraction={float(config.calibration_mixed_positive_fraction):.4f}, "
            f"pool_size={config.calibration_pair_pool_size}"
        ),
        (
            "test_pair_construction: "
            f"policy={config.test_pair_policy}, "
            f"target={config.test_pair_policy_target}, "
            f"mixed_positive_fraction={float(config.test_mixed_positive_fraction):.4f}, "
            f"pool_size={config.test_pair_pool_size}"
        ),
        f"factual_validation_exact_acc: {float(factual_metrics.get('exact_acc', 0.0)):.4f}",
        "",
    ]
    for bank in (train_bank, calibration_bank, test_bank):
        stats = dict(bank.pair_stats)
        split = str(bank.split)
        summary_lines.extend(
            [
                (
                    f"{split} pair bank | total_pairs={int(stats.get('total_pairs', 0))} "
                    f"| changed_any={int(stats.get('changed_any_count', 0))} "
                    f"| unchanged_any={int(stats.get('unchanged_any_count', 0))}"
                )
            ]
        )
        per_variable = dict(stats.get("per_variable", {}))
        for variable, variable_stats in per_variable.items():
            summary_lines.append(
                (
                    f"{split} pair bank [{variable}] | changed={int(variable_stats.get('changed_count', 0))} "
                    f"| unchanged={int(variable_stats.get('unchanged_count', 0))} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                )
            )
    summary_lines.append("")
    for method in config.methods:
        summary_lines.append(format_method_selection_summary(method_selections[method]))
        summary_lines.append("")
    summary_lines.append("Average Summary")
    for record in summary_records:
        summary_lines.append(
            f"{str(record['method']).upper()}: "
            f"exact={float(record['exact_acc']):.4f}, "
            f"shared={float(record['mean_shared_digits']):.4f}, "
            f"runtime_s={float(method_runtime_seconds.get(str(record['method']), 0.0)):.2f}"
        )
    candidate_sections = []
    for method in config.methods:
        candidate_section = format_method_candidate_sweep(method, method_payloads[method])
        if candidate_section:
            candidate_sections.append(candidate_section)
    if candidate_sections:
        summary_lines.extend(
            [
                "",
                "-" * 72,
                "",
                "Candidate Sweeps",
                "",
                "\n\n".join(candidate_sections),
            ]
        )
    return summary_lines, method_selections


def run_gradient_comparison_with_model(
    *,
    problem,
    model,
    backbone_meta: dict[str, object],
    device,
    config: GradientRunConfig,
) -> dict[str, object]:
    train_bank = build_pair_bank(
        problem,
        config.train_pair_size,
        config.seed + 201,
        "train",
        target_vars=tuple(config.target_vars),
        pair_policy=config.train_pair_policy,
        pair_policy_target=config.train_pair_policy_target,
        mixed_positive_fraction=config.train_mixed_positive_fraction,
        pair_pool_size=config.train_pair_pool_size,
    )
    calibration_bank = build_pair_bank(
        problem,
        config.calibration_pair_size,
        config.seed + 301,
        "calibration",
        target_vars=tuple(config.target_vars),
        pair_policy=config.calibration_pair_policy,
        pair_policy_target=config.calibration_pair_policy_target,
        mixed_positive_fraction=config.calibration_mixed_positive_fraction,
        pair_pool_size=config.calibration_pair_pool_size,
    )
    test_bank = build_pair_bank(
        problem,
        config.test_pair_size,
        config.seed + 401,
        "test",
        target_vars=tuple(config.target_vars),
        pair_policy=config.test_pair_policy,
        pair_policy_target=config.test_pair_policy_target,
        mixed_positive_fraction=config.test_mixed_positive_fraction,
        pair_pool_size=config.test_pair_pool_size,
    )

    method_payloads: dict[str, dict[str, object]] = {}
    method_runtime_seconds: dict[str, float] = {}
    all_records: list[dict[str, object]] = []
    for method_index, method in enumerate(config.methods, start=1):
        print(f"[{method_index}/{len(config.methods)}] Starting {method.upper()} gradient")
        method_start_time = perf_counter()
        payload = run_alignment_gradient_pipeline(
            model=model,
            fit_bank=train_bank,
            calibration_bank=calibration_bank,
            holdout_bank=test_bank,
            device=device,
            config=OTGradientConfig(
                method=method,
                batch_size=config.batch_size,
                resolution=config.resolution,
                alpha=config.fgw_alpha,
                target_vars=tuple(config.target_vars),
                policy_learning_rate=config.policy_learning_rate,
                policy_epochs=config.policy_epochs,
                policy_min_epochs=config.policy_min_epochs,
                policy_plateau_patience=config.policy_plateau_patience,
                policy_plateau_rel_delta=config.policy_plateau_rel_delta,
                policy_temperature=config.policy_temperature,
                policy_eval_interval=config.policy_eval_interval,
                fixed_top_k=config.fixed_top_k,
                fixed_lambda=config.fixed_lambda,
            ),
        )
        runtime_seconds = perf_counter() - method_start_time
        payload["runtime_seconds"] = float(runtime_seconds)
        method_payloads[method] = payload
        method_runtime_seconds[method] = float(runtime_seconds)
        all_records.extend(payload["results"])
        print(f"{method.upper()} gradient runtime: {float(runtime_seconds):.2f}s")
        print()

    summary_records = summarize_method_records(all_records)
    for record in summary_records:
        record["runtime_seconds"] = float(method_runtime_seconds.get(str(record["method"]), 0.0))
    summary_lines, method_selections = _build_gradient_summary_lines(
        config=config,
        device=device,
        backbone_meta=backbone_meta,
        train_bank=train_bank,
        calibration_bank=calibration_bank,
        test_bank=test_bank,
        method_payloads=method_payloads,
        method_runtime_seconds=method_runtime_seconds,
        summary_records=summary_records,
    )
    payload = {
        "seed": config.seed,
        "methods": list(config.methods),
        "checkpoint_path": str(config.checkpoint_path),
        "target_vars": list(config.target_vars),
        "pair_construction": {
            "train": {
                "pair_policy": config.train_pair_policy,
                "pair_policy_target": config.train_pair_policy_target,
                "mixed_positive_fraction": float(config.train_mixed_positive_fraction),
                "pair_pool_size": config.train_pair_pool_size,
            },
            "calibration": {
                "pair_policy": config.calibration_pair_policy,
                "pair_policy_target": config.calibration_pair_policy_target,
                "mixed_positive_fraction": float(config.calibration_mixed_positive_fraction),
                "pair_pool_size": config.calibration_pair_pool_size,
            },
            "test": {
                "pair_policy": config.test_pair_policy,
                "pair_policy_target": config.test_pair_policy_target,
                "mixed_positive_fraction": float(config.test_mixed_positive_fraction),
                "pair_pool_size": config.test_pair_pool_size,
            },
        },
        "backbone": backbone_meta,
        "banks": {
            "train": train_bank.metadata(),
            "calibration": calibration_bank.metadata(),
            "test": test_bank.metadata(),
        },
        "results": all_records,
        "method_summary": summary_records,
        "method_selections": method_selections,
        "method_runtime_seconds": method_runtime_seconds,
    }

    plot_paths = save_comparison_plots(payload, config.output_path, method_payloads=method_payloads)
    payload["plots"] = plot_paths
    payload["summary_path"] = str(config.summary_path)
    write_json(config.output_path, payload)
    write_text_report(config.summary_path, "\n".join(summary_lines))

    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    print(f"Backbone factual validation accuracy: {float(factual_metrics.get('exact_acc', 0.0)):.4f}")
    print_results_table(all_records, "Counterfactual Test Results")
    print_results_table(summary_records, "Method Average Summary")
    print(f"Wrote gradient comparison results to {Path(config.output_path).resolve()}")
    print(f"Wrote gradient comparison summary to {Path(config.summary_path).resolve()}")
    return payload


def main() -> None:
    problem = load_addition_problem(run_checks=True)
    device = resolve_device(DEVICE)
    seed_runs = []

    for index, seed in enumerate(SEEDS, start=1):
        checkpoint_path = Path(CHECKPOINT_PATH_TEMPLATE.format(seed=seed))
        seed_run_dir = RUN_DIR / f"seed_{seed}"
        print(f"[{index}/{len(SEEDS)}] seed={seed} | checkpoint={checkpoint_path}")
        model, backbone_meta, backbone_source = load_or_train_backbone(
            problem=problem,
            device=device,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        if backbone_source == "loaded":
            print_loaded_backbone_validation(backbone_meta)
        comparison = run_gradient_comparison_with_model(
            problem=problem,
            model=model,
            backbone_meta=backbone_meta,
            device=device,
            config=build_run_config(seed, checkpoint_path, seed_run_dir),
        )
        seed_runs.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "backbone_source": backbone_source,
                "comparison": comparison,
            }
        )
        print()

    if len(SEEDS) > 1:
        payload = build_seed_sweep_payload(seed_runs)
        payload["device"] = str(device)
        payload["checkpoint_path_template"] = CHECKPOINT_PATH_TEMPLATE
        payload["retrain_backbones"] = RETRAIN_BACKBONES
        payload["training_config"] = {
            "train_size": FACTUAL_TRAIN_SIZE,
            "validation_size": FACTUAL_VALIDATION_SIZE,
            "hidden_dims": list(HIDDEN_DIMS),
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
        }
        payload["comparison_config"] = {
            "methods": list(METHODS),
            "train_pair_size": TRAIN_PAIR_SIZE,
            "calibration_pair_size": CALIBRATION_PAIR_SIZE,
            "test_pair_size": TEST_PAIR_SIZE,
            "train_pair_policy": TRAIN_PAIR_POLICY,
            "train_pair_policy_target": TRAIN_PAIR_POLICY_TARGET,
            "train_mixed_positive_fraction": TRAIN_MIXED_POSITIVE_FRACTION,
            "train_pair_pool_size": TRAIN_PAIR_POOL_SIZE,
            "calibration_pair_policy": CALIBRATION_PAIR_POLICY,
            "calibration_pair_policy_target": CALIBRATION_PAIR_POLICY_TARGET,
            "calibration_mixed_positive_fraction": CALIBRATION_MIXED_POSITIVE_FRACTION,
            "calibration_pair_pool_size": CALIBRATION_PAIR_POOL_SIZE,
            "test_pair_policy": TEST_PAIR_POLICY,
            "test_pair_policy_target": TEST_PAIR_POLICY_TARGET,
            "test_mixed_positive_fraction": TEST_MIXED_POSITIVE_FRACTION,
            "test_pair_pool_size": TEST_PAIR_POOL_SIZE,
            "batch_size": BATCH_SIZE,
            "resolution": RESOLUTION,
            "fgw_alpha": FGW_ALPHA,
            "policy_learning_rate": POLICY_LEARNING_RATE,
            "policy_epochs": POLICY_EPOCHS,
            "policy_min_epochs": POLICY_MIN_EPOCHS,
            "policy_plateau_patience": POLICY_PLATEAU_PATIENCE,
            "policy_plateau_rel_delta": POLICY_PLATEAU_REL_DELTA,
            "policy_temperature": POLICY_TEMPERATURE,
            "policy_eval_interval": POLICY_EVAL_INTERVAL,
            "fixed_top_k": FIXED_TOP_K,
            "fixed_lambda": FIXED_LAMBDA,
            "selection_objective": "calibration_cross_entropy",
            "final_evaluation_policy": "hard_single_layer_top_k",
        }
        payload["summary_path"] = str(SUMMARY_PATH)
        payload["plots"] = save_seed_sweep_plots(payload, OUTPUT_PATH)
        write_json(OUTPUT_PATH, payload)
        write_text_report(SUMMARY_PATH, format_seed_sweep_summary(payload))

        print(f"Wrote gradient run results to {Path(OUTPUT_PATH).resolve()}")
        print(f"Wrote gradient run summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
