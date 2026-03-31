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
from equality_experiment.plots import DEFAULT_PASTEL_COLORS, METHOD_PASTEL_COLORS
from equality_experiment.reporting import write_text_report
from equality_experiment.runtime import ensure_parent_dir, resolve_device, write_json
from equality_experiment.scm import load_equality_problem


SEED = 42
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality"
CHECKPOINT_PATH = Path(f"models/equality_mlp_seed{SEED}.pt")
OUTPUT_PATH = RUN_DIR / "equality_run_results.json"
SUMMARY_PATH = RUN_DIR / "equality_run_summary.txt"
RETRAIN_BACKBONE = True

METHODS = ("ot", "das")
TARGET_VARS = ("WX", "YZ")
TRANSPORT_METHODS = tuple(method for method in METHODS if method in {"ot", "uot", "gw", "fgw"})
NON_TRANSPORT_METHODS = tuple(method for method in METHODS if method not in {"ot", "uot", "gw", "fgw"})

NUM_ENTITIES = 100
EMBEDDING_DIM = 4

FACTUAL_TRAIN_SIZE = 1048576
FACTUAL_VALIDATION_SIZE = 10000
HIDDEN_DIMS = (16, 16, 16)
LEARNING_RATE = 1e-3
EPOCHS = 3
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

# shared evaluation batch size (for training DAS or for calibrating )
BATCH_SIZE = 128

RESOLUTION = 1
FGW_ALPHA = 0.5
OT_EPSILONS = (0.05,)
OT_TAUS = (1.0,)
UOT_BETA_ABSTRACTS = (1.0,)
UOT_BETA_NEURALS = (1.0,)
SIGNATURE_MODES = ("prob_delta",)
OT_TOP_K_VALUES = tuple(range(1, 21))
OT_LAMBDAS = tuple(round(x * 0.1, 1) for x in range(1, 81))

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 1
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = (1, 4, 8, 12, 16)
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
    run_dir: Path,
) -> CompareExperimentConfig:
    """Build the equality comparison config."""
    return CompareExperimentConfig(
        seed=SEED,
        checkpoint_path=CHECKPOINT_PATH,
        output_path=run_dir / "equality_run_results.json",
        summary_path=run_dir / "equality_run_summary.txt",
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
        signature_mode=str(signature_mode),
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
        if "method" in source:
            source_label = (
                f"sweep eps={float(source.get('ot_epsilon', 0.0)):.3g}, "
                f"tau={float(source.get('ot_tau', 0.0)):.3g}"
            )
        else:
            source_label = "fixed_methods"
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


def _save_best_method_exact_plot(table_records: list[dict[str, object]], output_path: Path) -> str | None:
    """Save a grouped bar chart for the best per-method exact accuracies."""
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
            color=METHOD_PASTEL_COLORS.get(method, DEFAULT_PASTEL_COLORS[idx % len(DEFAULT_PASTEL_COLORS)]),
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
        f"train_pair_policy: {TRAIN_PAIR_POLICY}",
        f"train_pair_policy_target: {TRAIN_PAIR_POLICY_TARGET}",
        f"train_mixed_positive_fraction: {TRAIN_MIXED_POSITIVE_FRACTION}",
        f"train_pair_pool_size: {TRAIN_PAIR_POOL_SIZE}",
        f"calibration_pair_policy: {CALIBRATION_PAIR_POLICY}",
        f"calibration_pair_policy_target: {CALIBRATION_PAIR_POLICY_TARGET}",
        f"calibration_mixed_positive_fraction: {CALIBRATION_MIXED_POSITIVE_FRACTION}",
        f"calibration_pair_pool_size: {CALIBRATION_PAIR_POOL_SIZE}",
        f"test_pair_policy: {TEST_PAIR_POLICY}",
        f"test_pair_policy_target: {TEST_PAIR_POLICY_TARGET}",
        f"test_mixed_positive_fraction: {TEST_MIXED_POSITIVE_FRACTION}",
        f"test_pair_pool_size: {TEST_PAIR_POOL_SIZE}",
        f"batch_size: {BATCH_SIZE}",
        f"resolution: {RESOLUTION}",
        f"fgw_alpha: {FGW_ALPHA}",
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
    static_runs = []
    static_summary_sections = []
    if NON_TRANSPORT_METHODS:
        static_run_dir = RUN_DIR / "fixed_methods"
        print(f"[fixed] methods={', '.join(NON_TRANSPORT_METHODS)}")
        comparison = run_comparison_with_banks(
            model=model,
            backbone_meta=backbone_meta,
            device=device,
            config=build_compare_config(
                NON_TRANSPORT_METHODS,
                OT_EPSILONS[0],
                OT_TAUS[0],
                UOT_BETA_ABSTRACTS[0],
                UOT_BETA_NEURALS[0],
                SIGNATURE_MODES[0],
                static_run_dir,
            ),
            train_bank=train_bank,
            calibration_bank=calibration_bank,
            test_bank=test_bank,
        )
        static_summary_path = static_run_dir / "equality_run_summary.txt"
        static_runs.append(
            {
                "methods": list(NON_TRANSPORT_METHODS),
                "output_path": str(static_run_dir / "equality_run_results.json"),
                "summary_path": str(static_summary_path),
                "comparison": comparison,
            }
        )
        static_summary_text = Path(static_summary_path).read_text(encoding="utf-8").rstrip()
        static_summary_sections.append(
            "\n".join(
                [
                    f"Fixed methods: {', '.join(NON_TRANSPORT_METHODS)}",
                    "",
                    static_summary_text,
                ]
            )
        )

    epsilon_sweeps = []
    epsilon_summary_sections = []
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
            epsilon_tag = _format_epsilon_tag(ot_epsilon)
            tau_tag = _format_tau_tag(ot_tau)
            beta_abstract_tag = _format_beta_tag(uot_beta_abstract)
            beta_neural_tag = _format_beta_tag(uot_beta_neural)
            epsilon_run_dir = RUN_DIR / (
                f"{method}_sig_{signature_mode}_epsilon_{epsilon_tag}_tau_{tau_tag}"
                f"_betaa_{beta_abstract_tag}_betan_{beta_neural_tag}"
            )
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
                    epsilon_run_dir,
                ),
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                test_bank=test_bank,
            )
            epsilon_summary_path = epsilon_run_dir / "equality_run_summary.txt"
            epsilon_sweeps.append(
                {
                    "method": method,
                    "ot_epsilon": float(ot_epsilon),
                    "ot_tau": float(ot_tau),
                    "uot_beta_abstract": float(uot_beta_abstract),
                    "uot_beta_neural": float(uot_beta_neural),
                    "signature_mode": signature_mode,
                    "methods": [method],
                    "output_path": str(epsilon_run_dir / "equality_run_results.json"),
                    "summary_path": str(epsilon_summary_path),
                    "comparison": comparison,
                }
            )
            summary_text = Path(epsilon_summary_path).read_text(encoding="utf-8").rstrip()
            epsilon_summary_sections.append(
                "\n".join(
                    [
                        (
                            f"Transport sweep: method={method}, signature_mode={signature_mode}, "
                            f"epsilon={float(ot_epsilon):.6f}, tau={float(ot_tau):.6f}, "
                            f"uot_beta_abstract={float(uot_beta_abstract):.6f}, "
                            f"uot_beta_neural={float(uot_beta_neural):.6f}"
                        ),
                        "",
                        summary_text,
                    ]
                )
            )

    payload = {
        "device": str(device),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "retrain_backbone": RETRAIN_BACKBONE,
        "ot_epsilons": [float(value) for value in OT_EPSILONS],
        "ot_taus": [float(value) for value in OT_TAUS],
        "uot_beta_abstracts": [float(value) for value in UOT_BETA_ABSTRACTS],
        "uot_beta_neurals": [float(value) for value in UOT_BETA_NEURALS],
        "signature_modes": list(SIGNATURE_MODES),
        "target_vars": list(TARGET_VARS),
        "fixed_method_runs": static_runs,
        "epsilon_sweeps": epsilon_sweeps,
        "summary_path": str(SUMMARY_PATH),
    }
    best_method_table_records, best_method_table_text = _build_best_method_exact_table(static_runs, epsilon_sweeps)
    best_method_plot_path = _save_best_method_exact_plot(
        best_method_table_records,
        RUN_DIR / "best_method_exact_accuracy.png",
    )
    payload["best_method_exact_table"] = best_method_table_records
    payload["best_method_exact_plot"] = best_method_plot_path
    write_json(OUTPUT_PATH, payload)
    sections = [*static_summary_sections, *epsilon_summary_sections]
    write_text_report(
        SUMMARY_PATH,
        "\n".join(main_summary_lines)
        + "\n\n"
        + best_method_table_text
        + "\n\n"
        + (f"best_method_exact_plot: {best_method_plot_path}\n\n" if best_method_plot_path is not None else "")
        + ("\n\n" + ("=" * 80) + "\n\n").join(sections),
    )

    print(best_method_table_text)
    if best_method_plot_path is not None:
        print(f"Wrote best-method exact plot to {Path(best_method_plot_path).resolve()}")

    print(f"Wrote run results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote run summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
