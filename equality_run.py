import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
from equality_experiment.compare_runner import CompareExperimentConfig, run_comparison_with_banks
from equality_experiment.pair_bank import build_pair_bank
from equality_experiment.plots import get_method_color
from equality_experiment.reporting import format_method_selection_summary, write_text_report
from equality_experiment.runtime import ensure_parent_dir, resolve_device, write_json
from equality_experiment.scm import load_equality_problem


SEED = 1
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality"
CHECKPOINT_PATH = Path(f"models/equality_mlp_seed{SEED}.pt")
OUTPUT_PATH = RUN_DIR / "equality_run_results.json"
SUMMARY_PATH = RUN_DIR / "equality_run_summary.txt"
RETRAIN_BACKBONE = False

METHODS = ["ot"] #, "uot", "das"]
TARGET_VARS = ["WX", "YZ"]
TRANSPORT_METHODS = tuple(method for method in METHODS if method in {"ot", "uot", "gw", "fgw"})
NON_TRANSPORT_METHODS = tuple(method for method in METHODS if method not in {"ot", "uot", "gw", "fgw"})

NUM_ENTITIES = 100
EMBEDDING_DIM = 4

FACTUAL_TRAIN_SIZE = 1048576
FACTUAL_VALIDATION_SIZE = 10000
HIDDEN_DIMS = [16, 16, 16]
LEARNING_RATE = 1e-3
EPOCHS = 10
TRAIN_BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 1024

TRAIN_PAIR_SIZE = 1000
CALIBRATION_PAIR_SIZE = 1000
TEST_PAIR_SIZE = 1000

# `*_PAIR_POLICY_TARGET` can be one of:
#   - "any", "WX", "YZ", "both", "C1_only", "C2_only"
TRAIN_PAIR_POLICY = "mixed"
TRAIN_PAIR_POLICY_TARGET = "any"
TRAIN_MIXED_POSITIVE_FRACTION = 0.5
TRAIN_PAIR_POOL_SIZE = 2048

CALIBRATION_PAIR_POLICY = "mixed"
CALIBRATION_PAIR_POLICY_TARGET = "WX"
CALIBRATION_MIXED_POSITIVE_FRACTION = 1.0
CALIBRATION_PAIR_POOL_SIZE = 2048

TEST_PAIR_POLICY = "mixed"
TEST_PAIR_POLICY_TARGET = "WX"
TEST_MIXED_POSITIVE_FRACTION = 1.0
TEST_PAIR_POOL_SIZE = 2048

TARGETED_EVAL = True
TARGETED_PAIR_POLICY = "mixed"
TARGETED_MIXED_POSITIVE_FRACTION = 1.0
TARGETED_PAIR_POOL_SIZE = 2048

# shared evaluation batch size (for training DAS or for calibrating )
BATCH_SIZE = 128

RESOLUTION = 1
FGW_ALPHA = 0.5
TRANSPORT_SOLVER_BACKEND = "custom"  # "custom" or "pot"
OT_EPSILONS = [1.0, 3.0, 10.0, 15.0, 20.0]# [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
OT_TAUS = [1.0]
UOT_BETA_ABSTRACTS = [0.1, 1.0]
UOT_BETA_NEURALS = [0.1, 1.0]
SIGNATURE_MODES = ["prob_delta"]
OT_TOP_K_VALUES = list(range(1, 21))
OT_LAMBDAS = [round(x * 0.1, 1) for x in range(1, 81)]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = [1, 4, 8, 12, 16]
DAS_LAYERS = None


def build_train_config() -> EqualityTrainConfig:
    """Build the equality backbone training config."""
    return EqualityTrainConfig(
        seed=SEED,
        n_train=FACTUAL_TRAIN_SIZE,
        n_validation=FACTUAL_VALIDATION_SIZE,
        hidden_dims=tuple(HIDDEN_DIMS),
        abstract_variables=tuple(TARGET_VARS),
        learning_rate=LEARNING_RATE,
        train_epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
    )


def _format_epsilon_tag(epsilon: float) -> str:
    """Build a stable directory/file tag for one epsilon sweep value."""
    return f"{float(epsilon):.6f}".rstrip("0").rstrip(".")


def _format_tau_tag(tau: float) -> str:
    """Build a stable directory/file tag for one tau sweep value."""
    return f"{float(tau):.6f}".rstrip("0").rstrip(".")


def _format_beta_tag(beta: float) -> str:
    """Build a stable directory/file tag for one UOT beta sweep value."""
    return f"{float(beta):.6f}".rstrip("0").rstrip(".")


def build_compare_config(
    methods: tuple[str, ...],
    ot_epsilon: float,
    ot_tau: float,
    uot_beta_abstract: float,
    uot_beta_neural: float,
    signature_mode: str,
    output_path: Path,
    summary_path: Path,
) -> CompareExperimentConfig:
    """Build the equality comparison config."""
    return CompareExperimentConfig(
        seed=SEED,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=output_path,
        summary_path=summary_path,
        methods=tuple(methods),
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
        ot_epsilon=float(ot_epsilon),
        ot_tau=float(ot_tau),
        uot_beta_abstract=float(uot_beta_abstract),
        uot_beta_neural=float(uot_beta_neural),
        transport_solver_backend=TRANSPORT_SOLVER_BACKEND,
        signature_mode=str(signature_mode),
        save_outputs=False,
        save_plots=False,
        ot_top_k_values=OT_TOP_K_VALUES,
        ot_lambdas=tuple(OT_LAMBDAS),
        das_max_epochs=DAS_MAX_EPOCHS,
        das_min_epochs=DAS_MIN_EPOCHS,
        das_plateau_patience=DAS_PLATEAU_PATIENCE,
        das_plateau_rel_delta=DAS_PLATEAU_REL_DELTA,
        das_learning_rate=DAS_LEARNING_RATE,
        das_subspace_dims=DAS_SUBSPACE_DIMS,
        das_layers=DAS_LAYERS,
    )


def _build_best_method_exact_table(
    static_runs: list[dict[str, object]],
    epsilon_sweeps: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """Select the best run per method and format a compact per-variable exact table."""
    candidate_runs = [*static_runs, *epsilon_sweeps]
    best_by_method: dict[str, dict[str, object]] = {}
    for candidate in candidate_runs:
        comparison = dict(candidate.get("comparison", {}))
        method_summary = list(comparison.get("method_summary", []))
        for summary_record in method_summary:
            method = str(summary_record["method"])
            exact_acc = float(summary_record.get("exact_acc", 0.0))
            current = best_by_method.get(method)
            if current is None or exact_acc > float(current["summary_record"].get("exact_acc", 0.0)):
                best_by_method[method] = {
                    "summary_record": summary_record,
                    "comparison": comparison,
                    "candidate": candidate,
                }

    table_records: list[dict[str, object]] = []
    for method, bundle in sorted(best_by_method.items()):
        comparison = dict(bundle["comparison"])
        candidate = dict(bundle["candidate"])
        method_records = [
            dict(record)
            for record in comparison.get("results", [])
            if str(record.get("method")) == method
        ]
        method_records.sort(key=lambda record: TARGET_VARS.index(str(record["variable"])))
        row = {
            "method": method,
            "average_exact_acc": float(bundle["summary_record"].get("exact_acc", 0.0)),
            "records": method_records,
            "source": candidate,
        }
        table_records.append(row)

    if not table_records:
        return [], "Best Per-Method Counterfactual Accuracy\n(no records)"

    variable_columns = [str(variable) for variable in TARGET_VARS]
    method_width = max(len("method"), max(len(str(row["method"]).upper()) for row in table_records))
    variable_widths = {
        variable: max(len(variable), 7)
        for variable in variable_columns
    }
    average_width = max(len("average"), 7)
    source_width = max(len("source"), 18)
    header = (
        f"{'method':<{method_width}}  "
        + "  ".join(f"{variable:>{variable_widths[variable]}}" for variable in variable_columns)
        + f"  {'average':>{average_width}}"
        + f"  {'source':<{source_width}}"
    )
    lines = [
        "Best Per-Method Counterfactual Accuracy",
        header,
        "-" * len(header),
    ]
    for row in table_records:
        exact_by_variable = {
            str(record["variable"]): float(record.get("exact_acc", 0.0))
            for record in row["records"]
        }
        source = dict(row["source"])
        if str(source.get("source_type")) == "fixed":
            source_label = "fixed"
        elif "method" in source:
            source_label = (
                f"sweep eps={float(source.get('ot_epsilon', 0.0)):.3g}, "
                f"tau={float(source.get('ot_tau', 0.0)):.3g}"
            )
        else:
            source_label = "fixed"
        lines.append(
            f"{str(row['method']).upper():<{method_width}}  "
            + "  ".join(
                f"{exact_by_variable.get(variable, float('nan')):>{variable_widths[variable]}.4f}"
                for variable in variable_columns
            )
            + f"  {float(row['average_exact_acc']):>{average_width}.4f}"
            + f"  {source_label:<{source_width}}"
        )
    return table_records, "\n".join(lines)


def _build_transport_config_stem(
    *,
    method: str,
    signature_mode: str,
    ot_epsilon: float,
    ot_tau: float,
    uot_beta_abstract: float,
    uot_beta_neural: float,
) -> str:
    """Build a stable file stem for one transport sweep configuration."""
    epsilon_tag = _format_epsilon_tag(ot_epsilon)
    tau_tag = _format_tau_tag(ot_tau)
    stem = f"sig_{signature_mode}_epsilon_{epsilon_tag}_tau_{tau_tag}"
    if method == "uot":
        beta_abstract_tag = _format_beta_tag(uot_beta_abstract)
        beta_neural_tag = _format_beta_tag(uot_beta_neural)
        stem += f"_betaa_{beta_abstract_tag}_betan_{beta_neural_tag}"
    return stem


def _build_transport_method_summary(method: str, sweep_records: list[dict[str, object]]) -> str:
    """Format one per-method summary with best-result details plus all config accuracies."""
    sorted_records = sorted(
        sweep_records,
        key=lambda record: float(
            next(
                (
                    summary.get("exact_acc", 0.0)
                    for summary in dict(record.get("comparison", {})).get("method_summary", [])
                    if str(summary.get("method")) == method
                ),
                0.0,
            )
        ),
        reverse=True,
    )
    lines = [f"{method.upper()} Sweep Summary", ""]
    if not sorted_records:
        lines.append("(no records)")
        return "\n".join(lines)

    best = sorted_records[0]
    best_comparison = dict(best.get("comparison", {}))
    best_summary = next(
        (
            dict(summary)
            for summary in best_comparison.get("method_summary", [])
            if str(summary.get("method")) == method
        ),
        {},
    )
    best_records = [
        dict(record)
        for record in best_comparison.get("results", [])
        if str(record.get("method")) == method
    ]
    best_exact_by_variable = {
        str(record["variable"]): float(record.get("exact_acc", 0.0))
        for record in best_records
    }
    lines.extend(
        [
            "Best Hyperparameters",
            f"signature_mode: {best.get('signature_mode')}",
            f"ot_epsilon: {float(best.get('ot_epsilon', 0.0)):.6f}",
            f"ot_tau: {float(best.get('ot_tau', 0.0)):.6f}",
        ]
    )
    if method == "uot":
        lines.extend(
            [
                f"uot_beta_abstract: {float(best.get('uot_beta_abstract', 0.0)):.6f}",
                f"uot_beta_neural: {float(best.get('uot_beta_neural', 0.0)):.6f}",
            ]
        )
    lines.append(f"average_exact_acc: {float(best_summary.get('exact_acc', 0.0)):.4f}")
    for variable in TARGET_VARS:
        lines.append(f"{variable}_exact_acc: {float(best_exact_by_variable.get(variable, 0.0)):.4f}")
    method_selection = dict(best_comparison.get("method_selections", {}).get(method, {}))
    if method_selection:
        lines.extend(
            [
                "",
                "Selected Intervention",
                format_method_selection_summary(method_selection),
            ]
        )
    lines.extend(["", "All Config Accuracies"])
    for record in sorted_records:
        comparison = dict(record.get("comparison", {}))
        method_selection = dict(comparison.get("method_selections", {}).get(method, {}))
        failed = bool(method_selection.get("failed", False))
        summary = next(
            (
                dict(item)
                for item in comparison.get("method_summary", [])
                if str(item.get("method")) == method
            ),
            {},
        )
        method_records = [
            dict(item)
            for item in comparison.get("results", [])
            if str(item.get("method")) == method
        ]
        exact_by_variable = {
            str(item["variable"]): float(item.get("exact_acc", 0.0))
            for item in method_records
        }
        bits = [
            f"avg={float(summary.get('exact_acc', 0.0)):.4f}",
            f"sig={record.get('signature_mode')}",
            f"epsilon={float(record.get('ot_epsilon', 0.0)):.6f}",
            f"tau={float(record.get('ot_tau', 0.0)):.6f}",
        ]
        if method == "uot":
            bits.extend(
                [
                    f"betaa={float(record.get('uot_beta_abstract', 0.0)):.6f}",
                    f"betan={float(record.get('uot_beta_neural', 0.0)):.6f}",
                ]
            )
        bits.extend(
            [
                f"{variable}={float(exact_by_variable.get(variable, 0.0)):.4f}"
                for variable in TARGET_VARS
            ]
        )
        if failed:
            reason = method_selection.get("failure_reason", "transport_failed")
            bits.append(f"FAILED={reason}")
        lines.append(", ".join(bits))
    return "\n".join(lines)


def _best_record_for_method(method: str, candidate_records: list[dict[str, object]]) -> dict[str, object] | None:
    """Select the highest-average-exact run for one method."""
    best_record = None
    best_score = float("-inf")
    for record in candidate_records:
        comparison = dict(record.get("comparison", {}))
        summary = next(
            (
                dict(item)
                for item in comparison.get("method_summary", [])
                if str(item.get("method")) == method
            ),
            None,
        )
        if summary is None:
            continue
        score = float(summary.get("exact_acc", 0.0))
        if score > best_score:
            best_score = score
            best_record = record
    return best_record


def _save_best_method_exact_plot(table_records: list[dict[str, object]], output_path: Path) -> str | None:
    """Save one top-level grouped bar chart for the best per-method exact accuracies."""
    if not table_records:
        return None
    ensure_parent_dir(output_path)
    methods = [str(row["method"]).upper() for row in table_records]
    variables = [str(variable) for variable in TARGET_VARS]
    x = np.arange(len(variables))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for idx, row in enumerate(table_records):
        exact_by_variable = {
            str(record["variable"]): float(record.get("exact_acc", 0.0))
            for record in row["records"]
        }
        y = [exact_by_variable.get(variable, np.nan) for variable in variables]
        method = str(row["method"]).lower()
        ax.bar(
            x + (idx - (len(methods) - 1) / 2.0) * width,
            y,
            width=width,
            color=get_method_color(method, idx),
            edgecolor="#6b7280",
            linewidth=0.8,
            label=method.upper(),
        )
    ax.set_xticks(x, variables)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Counterfactual Accuracy")
    ax.set_title("Best per-method accuracy by target variable")
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    problem = load_equality_problem(
        run_checks=True,
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        seed=SEED,
    )
    device = resolve_device(DEVICE)
    train_config = build_train_config()

    if RETRAIN_BACKBONE or not CHECKPOINT_PATH.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
        )
    else:
        model, _, backbone_meta = load_backbone(
            problem=problem,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            train_config=train_config,
        )

    train_bank = build_pair_bank(
        problem,
        TRAIN_PAIR_SIZE,
        SEED + 201,
        "train",
        target_vars=tuple(TARGET_VARS),
        pair_policy=TRAIN_PAIR_POLICY,
        pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        pair_pool_size=TRAIN_PAIR_POOL_SIZE,
    )
    if TARGETED_EVAL:
        calibration_bank = {}
        test_bank = {}
        for idx, variable in enumerate(TARGET_VARS):
            calibration_bank[variable] = build_pair_bank(
                problem,
                CALIBRATION_PAIR_SIZE,
                SEED + 301 + idx,
                "calibration",
                target_vars=tuple(TARGET_VARS),
                pair_policy=TARGETED_PAIR_POLICY,
                pair_policy_target=variable,
                mixed_positive_fraction=TARGETED_MIXED_POSITIVE_FRACTION,
                pair_pool_size=TARGETED_PAIR_POOL_SIZE,
            )
            test_bank[variable] = build_pair_bank(
                problem,
                TEST_PAIR_SIZE,
                SEED + 401 + idx,
                "test",
                target_vars=tuple(TARGET_VARS),
                pair_policy=TARGETED_PAIR_POLICY,
                pair_policy_target=variable,
                mixed_positive_fraction=TARGETED_MIXED_POSITIVE_FRACTION,
                pair_pool_size=TARGETED_PAIR_POOL_SIZE,
            )
    else:
        calibration_bank = build_pair_bank(
            problem,
            CALIBRATION_PAIR_SIZE,
            SEED + 301,
            "calibration",
            target_vars=tuple(TARGET_VARS),
            pair_policy=CALIBRATION_PAIR_POLICY,
            pair_policy_target=CALIBRATION_PAIR_POLICY_TARGET,
            mixed_positive_fraction=CALIBRATION_MIXED_POSITIVE_FRACTION,
            pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
        )
        test_bank = build_pair_bank(
            problem,
            TEST_PAIR_SIZE,
            SEED + 401,
            "test",
            target_vars=tuple(TARGET_VARS),
            pair_policy=TEST_PAIR_POLICY,
            pair_policy_target=TEST_PAIR_POLICY_TARGET,
            mixed_positive_fraction=TEST_MIXED_POSITIVE_FRACTION,
            pair_pool_size=TEST_PAIR_POOL_SIZE,
        )

    def _pair_bank_summary_lines(bank_or_banks) -> list[str]:
        if isinstance(bank_or_banks, dict):
            lines = []
            for variable, bank in bank_or_banks.items():
                stats = dict(bank.pair_stats)
                split = f"{bank.split}:{variable}"
                lines.append(
                    f"{split} pair bank | total_pairs={int(stats.get('total_pairs', 0))} "
                    f"| changed_any={int(stats.get('changed_any_count', 0))} "
                    f"| unchanged_any={int(stats.get('unchanged_any_count', 0))}"
                )
                per_variable = dict(stats.get("per_variable", {}))
                for stat_variable, variable_stats in per_variable.items():
                    lines.append(
                        f"{split} pair bank [{stat_variable}] | changed={int(variable_stats.get('changed_count', 0))} "
                        f"| unchanged={int(variable_stats.get('unchanged_count', 0))} "
                        f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                    )
            return lines
        stats = dict(bank_or_banks.pair_stats)
        split = str(bank_or_banks.split)
        lines = [
            (
                f"{split} pair bank | total_pairs={int(stats.get('total_pairs', 0))} "
                f"| changed_any={int(stats.get('changed_any_count', 0))} "
                f"| unchanged_any={int(stats.get('unchanged_any_count', 0))}"
            )
        ]
        per_variable = dict(stats.get("per_variable", {}))
        for variable, variable_stats in per_variable.items():
            lines.append(
                (
                    f"{split} pair bank [{variable}] | changed={int(variable_stats.get('changed_count', 0))} "
                    f"| unchanged={int(variable_stats.get('unchanged_count', 0))} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                )
            )
        return lines

    main_summary_lines = [
        "Hierarchical Equality Run Summary",
        f"device: {device}",
        f"methods: {', '.join(METHODS)}",
        f"target_vars: {', '.join(TARGET_VARS)}",
        f"num_entities: {NUM_ENTITIES}",
        f"embedding_dim: {EMBEDDING_DIM}",
        (
            "pair_sizes: "
            f"train={TRAIN_PAIR_SIZE}, "
            f"calibration={CALIBRATION_PAIR_SIZE}, "
            f"test={TEST_PAIR_SIZE}"
        ),
        f"targeted_eval: {TARGETED_EVAL}",
        f"train_pair_policy: {TRAIN_PAIR_POLICY}",
        f"train_pair_policy_target: {TRAIN_PAIR_POLICY_TARGET}",
        f"train_mixed_positive_fraction: {TRAIN_MIXED_POSITIVE_FRACTION}",
        f"train_pair_pool_size: {TRAIN_PAIR_POOL_SIZE}",
        *(
            [
                f"targeted_pair_policy: {TARGETED_PAIR_POLICY}",
                "targeted_pair_policy_target: per-variable positive pairs",
                f"targeted_mixed_positive_fraction: {TARGETED_MIXED_POSITIVE_FRACTION}",
                f"targeted_pair_pool_size: {TARGETED_PAIR_POOL_SIZE}",
            ]
            if TARGETED_EVAL
            else [
                f"calibration_pair_policy: {CALIBRATION_PAIR_POLICY}",
                f"calibration_pair_policy_target: {CALIBRATION_PAIR_POLICY_TARGET}",
                f"calibration_mixed_positive_fraction: {CALIBRATION_MIXED_POSITIVE_FRACTION}",
                f"calibration_pair_pool_size: {CALIBRATION_PAIR_POOL_SIZE}",
                f"test_pair_policy: {TEST_PAIR_POLICY}",
                f"test_pair_policy_target: {TEST_PAIR_POLICY_TARGET}",
                f"test_mixed_positive_fraction: {TEST_MIXED_POSITIVE_FRACTION}",
                f"test_pair_pool_size: {TEST_PAIR_POOL_SIZE}",
            ]
        ),
        f"batch_size: {BATCH_SIZE}",
        f"resolution: {RESOLUTION}",
        f"fgw_alpha: {FGW_ALPHA}",
        f"transport_solver_backend: {TRANSPORT_SOLVER_BACKEND}",
        "ot_epsilons: " + ", ".join(f"{float(value):.6f}" for value in OT_EPSILONS),
        "ot_taus: " + ", ".join(f"{float(value):.6f}" for value in OT_TAUS),
        "uot_beta_abstracts: " + ", ".join(f"{float(value):.6f}" for value in UOT_BETA_ABSTRACTS),
        "uot_beta_neurals: " + ", ".join(f"{float(value):.6f}" for value in UOT_BETA_NEURALS),
        "signature_modes: " + ", ".join(SIGNATURE_MODES),
        "ot_top_k_values: " + ", ".join(str(int(value)) for value in OT_TOP_K_VALUES),
        "ot_lambdas: " + ", ".join(f"{float(value):.6f}" for value in OT_LAMBDAS),
        f"das_max_epochs: {DAS_MAX_EPOCHS}",
        f"das_min_epochs: {DAS_MIN_EPOCHS}",
        f"das_plateau_patience: {DAS_PLATEAU_PATIENCE}",
        f"das_plateau_rel_delta: {DAS_PLATEAU_REL_DELTA}",
        f"das_learning_rate: {DAS_LEARNING_RATE}",
        (
            "das_subspace_dims: "
            + ("None" if DAS_SUBSPACE_DIMS is None else ", ".join(str(int(value)) for value in DAS_SUBSPACE_DIMS))
        ),
        f"das_layers: {DAS_LAYERS}",
        "",
    ]
    main_summary_lines.extend(_pair_bank_summary_lines(train_bank))
    main_summary_lines.extend(_pair_bank_summary_lines(calibration_bank))
    main_summary_lines.extend(_pair_bank_summary_lines(test_bank))
    main_summary_lines.append("")
    static_runs = []
    if NON_TRANSPORT_METHODS:
        for method in NON_TRANSPORT_METHODS:
            print(f"[fixed] method={method}")
            method_dir = RUN_DIR / method
            comparison = run_comparison_with_banks(
                model=model,
                backbone_meta=backbone_meta,
                device=device,
                config=build_compare_config(
                    (method,),
                    OT_EPSILONS[0],
                    OT_TAUS[0],
                    UOT_BETA_ABSTRACTS[0],
                    UOT_BETA_NEURALS[0],
                    SIGNATURE_MODES[0],
                    method_dir / f"{method}_results.json",
                    method_dir / f"{method}_summary.txt",
                ),
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                test_bank=test_bank,
            )
            static_runs.append(
                {
                    "method": method,
                    "methods": [method],
                    "source_type": "fixed",
                    "comparison": comparison,
                }
            )

    epsilon_sweeps = []
    if TRANSPORT_METHODS:
        method_sweep_points: list[tuple[str, float, float, float, float, str]] = []
        for method in TRANSPORT_METHODS:
            beta_abstracts = UOT_BETA_ABSTRACTS if method == "uot" else (UOT_BETA_ABSTRACTS[0],)
            beta_neurals = UOT_BETA_NEURALS if method == "uot" else (UOT_BETA_NEURALS[0],)
            signature_modes = SIGNATURE_MODES if method in {"ot", "uot", "gw", "fgw"} else (SIGNATURE_MODES[0],)
            for signature_mode in signature_modes:
                for ot_epsilon in OT_EPSILONS:
                    for ot_tau in OT_TAUS:
                        for uot_beta_abstract in beta_abstracts:
                            for uot_beta_neural in beta_neurals:
                                method_sweep_points.append(
                                    (
                                        method,
                                        float(ot_epsilon),
                                        float(ot_tau),
                                        float(uot_beta_abstract),
                                        float(uot_beta_neural),
                                        str(signature_mode),
                                    )
                                )

        for sweep_index, (method, ot_epsilon, ot_tau, uot_beta_abstract, uot_beta_neural, signature_mode) in enumerate(method_sweep_points, start=1):
            method_run_dir = RUN_DIR / method
            config_stem = _build_transport_config_stem(
                method=method,
                signature_mode=signature_mode,
                ot_epsilon=ot_epsilon,
                ot_tau=ot_tau,
                uot_beta_abstract=uot_beta_abstract,
                uot_beta_neural=uot_beta_neural,
            )
            epsilon_output_path = method_run_dir / f"{config_stem}_results.json"
            epsilon_summary_path = method_run_dir / f"{config_stem}_summary.txt"
            print(
                f"[sweep {sweep_index}/{len(method_sweep_points)}] "
                f"method={method} "
                f"| signature_mode={signature_mode} "
                f"| ot_epsilon={float(ot_epsilon):.6f} "
                f"| ot_tau={float(ot_tau):.6f} "
                f"| uot_beta_abstract={float(uot_beta_abstract):.6f} "
                f"| uot_beta_neural={float(uot_beta_neural):.6f}"
            )
            comparison = run_comparison_with_banks(
                model=model,
                backbone_meta=backbone_meta,
                device=device,
                config=build_compare_config(
                    (method,),
                    ot_epsilon,
                    ot_tau,
                    uot_beta_abstract,
                    uot_beta_neural,
                    signature_mode,
                    epsilon_output_path,
                    epsilon_summary_path,
                ),
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                test_bank=test_bank,
            )
            epsilon_sweeps.append(
                {
                    "method": method,
                    "ot_epsilon": float(ot_epsilon),
                    "ot_tau": float(ot_tau),
                    "uot_beta_abstract": float(uot_beta_abstract),
                    "uot_beta_neural": float(uot_beta_neural),
                    "signature_mode": signature_mode,
                    "config_stem": config_stem,
                    "methods": [method],
                    "comparison": comparison,
                }
            )

    best_method_runs = {}
    for record in static_runs:
        method = str(record.get("method"))
        comparison = dict(record.get("comparison", {}))
        method_dir = RUN_DIR / method
        result_path = method_dir / f"{method}_results.json"
        summary_path = method_dir / f"{method}_summary.txt"
        comparison["summary_path"] = str(summary_path)
        write_json(result_path, comparison)
        summary_text = f"{method.upper()} Summary"
        if "method_selections" in comparison:
            summary_entry = dict(comparison["method_selections"].get(method, {}))
            if summary_entry:
                summary_text = format_method_selection_summary(summary_entry)
        write_text_report(summary_path, summary_text)
        best_method_runs[method] = {
            "method": method,
            "source_type": "fixed",
            "output_path": str(result_path),
            "summary_path": str(summary_path),
            "comparison": comparison,
        }

    payload = {
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "retrain_backbone": RETRAIN_BACKBONE,
        "targeted_eval": TARGETED_EVAL,
        "ot_epsilons": [float(value) for value in OT_EPSILONS],
        "ot_taus": [float(value) for value in OT_TAUS],
        "uot_beta_abstracts": [float(value) for value in UOT_BETA_ABSTRACTS],
        "uot_beta_neurals": [float(value) for value in UOT_BETA_NEURALS],
        "signature_modes": list(SIGNATURE_MODES),
        "target_vars": list(TARGET_VARS),
        "best_method_runs": {},
        "summary_path": str(SUMMARY_PATH),
    }
    for method in TRANSPORT_METHODS:
        method_records = [record for record in epsilon_sweeps if str(record.get("method")) == method]
        if not method_records:
            continue
        best_record = _best_record_for_method(method, method_records)
        if best_record is None:
            continue
        method_dir = RUN_DIR / method
        result_path = method_dir / f"{method}_results.json"
        summary_path = method_dir / f"{method}_summary.txt"
        comparison = dict(best_record.get("comparison", {}))
        comparison["summary_path"] = str(summary_path)
        write_json(result_path, comparison)
        summary_text = _build_transport_method_summary(method, method_records)
        write_text_report(summary_path, summary_text)
        best_method_runs[method] = {
            "method": method,
            "source_type": "sweep",
            "ot_epsilon": float(best_record.get("ot_epsilon", 0.0)),
            "ot_tau": float(best_record.get("ot_tau", 0.0)),
            "uot_beta_abstract": float(best_record.get("uot_beta_abstract", 0.0)),
            "uot_beta_neural": float(best_record.get("uot_beta_neural", 0.0)),
            "signature_mode": best_record.get("signature_mode"),
            "output_path": str(result_path),
            "summary_path": str(summary_path),
            "comparison": comparison,
        }
    payload["best_method_runs"] = best_method_runs
    best_method_table_records, best_method_table_text = _build_best_method_exact_table(
        list(best_method_runs.values()),
        [],
    )
    best_method_plot_path = _save_best_method_exact_plot(
        best_method_table_records,
        RUN_DIR / "best_method_exact_accuracy.png",
    )
    payload["best_method_exact_table"] = best_method_table_records
    payload["best_method_exact_plot"] = best_method_plot_path
    best_method_sections = []
    for method, record in sorted(best_method_runs.items()):
        comparison = dict(record.get("comparison", {}))
        method_selection = dict(comparison.get("method_selections", {}).get(method, {}))
        if not method_selection:
            continue
        section_lines = [f"{method.upper()} Best Result"]
        if str(record.get("source_type")) == "sweep":
            section_lines.extend(
                [
                    f"signature_mode: {record.get('signature_mode')}",
                    f"ot_epsilon: {float(record.get('ot_epsilon', 0.0)):.6f}",
                    f"ot_tau: {float(record.get('ot_tau', 0.0)):.6f}",
                ]
            )
            if method == "uot":
                section_lines.extend(
                    [
                        f"uot_beta_abstract: {float(record.get('uot_beta_abstract', 0.0)):.6f}",
                        f"uot_beta_neural: {float(record.get('uot_beta_neural', 0.0)):.6f}",
                    ]
                )
        section_lines.extend(["", format_method_selection_summary(method_selection)])
        best_method_sections.append("\n".join(section_lines))
    write_json(OUTPUT_PATH, payload)
    write_text_report(
        SUMMARY_PATH,
        "\n".join(main_summary_lines)
        + "\n\n"
        + best_method_table_text
        + "\n\n"
        + (f"best_method_exact_plot: {best_method_plot_path}\n\n" if best_method_plot_path is not None else "")
        + ("\n\n" + ("=" * 72) + "\n\n").join(best_method_sections),
    )

    print(best_method_table_text)
    if best_method_plot_path is not None:
        print(f"Wrote best-method exact plot to {Path(best_method_plot_path).resolve()}")
    print(f"Wrote run results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote run summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
