import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import FuncNorm
from matplotlib.patches import Patch
from PIL import Image

from equality_experiment.backbone import EqualityTrainConfig, load_backbone, train_backbone
from equality_experiment.compare_runner import CompareExperimentConfig, run_comparison_with_banks
from equality_experiment.pair_bank import PairBank, build_pair_bank
from equality_experiment.plots import get_method_color
from equality_experiment.reporting import format_method_selection_summary, write_text_report
from equality_experiment.runtime import ensure_parent_dir, resolve_device, write_json
from equality_experiment.scm import load_equality_problem


SEEDS = [i for i in range(1,11)]
DEVICE = "cpu"
RUN_TIMESTAMP = os.environ.get("RESULTS_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = Path("results") / f"{RUN_TIMESTAMP}_equality"
OUTPUT_PATH = RUN_DIR / "equality_run_results.json"
SUMMARY_PATH = RUN_DIR / "equality_run_summary.txt"
RETRAIN_BACKBONE = False

METHODS = ["ot", "das"]
TARGET_VARS = ["WX", "YZ"]
TRANSPORT_METHODS = tuple(method for method in METHODS if method in {"ot", "uot", "gw", "fgw"})
NON_TRANSPORT_METHODS = tuple(method for method in METHODS if method not in {"ot", "uot", "gw", "fgw"})
HEQ_METHOD_ORDER = ("ot", "uot", "das")
HEQ_METHOD_LABELS = {"ot": "OT", "uot": "UOT", "das": "DAS"}
HEQ_METHOD_COLORS = {"ot": "#59a14f", "uot": "#76b7b2", "das": "#e15759"}

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

CALIBRATION_PAIR_POOL_SIZE = 2048
# Calibration strategy options:
# - "shared_wx_positive": one shared calibration bank containing only pairs where WX is sensitive.
# - "shared_yz_positive": one shared calibration bank containing only pairs where YZ is sensitive.
# - "shared_any_positive": one shared calibration bank containing only pairs where at least one target variable is sensitive.
# - "shared_any_mixed50": one shared calibration bank targeted to any sensitivity with a 50/50 sensitive-invariant mix.
# - "shared_both_positive": one shared calibration bank containing only pairs where both WX and YZ are sensitive.
# - "shared_balanced_wx_yz_only": one shared calibration bank made by concatenating half WX-sensitive/YZ-invariant pairs and half YZ-sensitive/WX-invariant pairs.
# - "separate_variable_positive": separate calibration banks for WX and YZ, each containing pairs where that variable is sensitive.
# - "separate_variable_only": separate calibration banks containing WX-sensitive/YZ-invariant pairs and YZ-sensitive/WX-invariant pairs, respectively.
# - "sensitive_test_eval": separate calibration banks for WX and YZ using the sensitive-test settings below.
CALIBRATION_STRATEGY = "shared_balanced_wx_yz_only"  # "shared_any_mixed50"

SENSITIVE_TEST_EVAL = True
SENSITIVE_TEST_PAIR_POLICY = "mixed"
SENSITIVE_TEST_MIXED_POSITIVE_FRACTION = 1.0
SENSITIVE_TEST_PAIR_POOL_SIZE = 2048
INVARIANT_TEST_EVAL = True
INVARIANT_TEST_PAIR_POLICY = "mixed"
INVARIANT_TEST_MIXED_POSITIVE_FRACTION = 0.0
INVARIANT_TEST_PAIR_POOL_SIZE = 2048

# shared evaluation batch size (for training DAS or for calibrating )
BATCH_SIZE = 128

RESOLUTION = 1
FGW_ALPHA = 0.5
TRANSPORT_SOLVER_BACKEND = "custom"  # "custom" or "pot"
OT_EPSILONS = [i for i in range(1,31)] # [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
OT_TAUS = [1.0]
UOT_BETA_ABSTRACTS = [1e6]
UOT_BETA_NEURALS = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
SIGNATURE_MODES = ["prob_delta"]
OT_TOP_K_VALUES = list(range(1, 21))
# OT_LAMBDAS = [round(x * 0.1, 1) for x in range(1, 81)]
OT_LAMBDAS = [i for i in range(1, 81)]

DAS_MAX_EPOCHS = 1000
DAS_MIN_EPOCHS = 5
DAS_PLATEAU_PATIENCE = 3
DAS_PLATEAU_REL_DELTA = 1e-2
DAS_LEARNING_RATE = 1e-3
DAS_SUBSPACE_DIMS = [i+1 for i in range(16)]
DAS_LAYERS = None


def build_train_config() -> EqualityTrainConfig:
    """Build the equality backbone training config."""
    return EqualityTrainConfig(
        seed=SEEDS[0],
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


def build_train_config_for_seed(seed: int) -> EqualityTrainConfig:
    """Build the equality backbone training config for one seed."""
    config = build_train_config()
    return EqualityTrainConfig(
        seed=int(seed),
        n_train=config.n_train,
        n_validation=config.n_validation,
        hidden_dims=config.hidden_dims,
        abstract_variables=config.abstract_variables,
        learning_rate=config.learning_rate,
        train_epochs=config.train_epochs,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        num_entities=config.num_entities,
        embedding_dim=config.embedding_dim,
        num_classes=config.num_classes,
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


def _pair_stats_from_bank(bank: PairBank) -> dict[str, object]:
    """Recompute pair stats for a concatenated bank."""
    per_variable = {}
    for variable in bank.pair_policy_vars:
        changed = int(bank.changed_by_var[variable].to(torch.int64).sum().item())
        total = int(bank.size)
        per_variable[variable] = {
            "changed_count": changed,
            "unchanged_count": total - changed,
            "changed_rate": float(changed / total) if total else 0.0,
        }
    changed_any = int(bank.changed_any.to(torch.int64).sum().item())
    total = int(bank.size)
    return {
        "total_pairs": total,
        "changed_any_count": changed_any,
        "unchanged_any_count": total - changed_any,
        "changed_any_rate": float(changed_any / total) if total else 0.0,
        "per_variable": per_variable,
    }


def _concat_pair_banks(
    banks: list[PairBank],
    *,
    split: str,
    seed: int,
    pair_policy: str,
    pair_policy_target: str,
    mixed_positive_fraction: float,
) -> PairBank:
    """Concatenate multiple compatible pair banks into one logical bank."""
    if not banks:
        raise ValueError("Expected at least one bank to concatenate")
    first = banks[0]
    for bank in banks[1:]:
        if bank.target_vars != first.target_vars:
            raise ValueError("Mismatched target_vars across banks")
        if bank.pair_policy_vars != first.pair_policy_vars:
            raise ValueError("Mismatched pair_policy_vars across banks")

    merged = PairBank(
        split=split,
        seed=seed,
        base_rows=torch.cat([bank.base_rows for bank in banks], dim=0),
        source_rows=torch.cat([bank.source_rows for bank in banks], dim=0),
        base_inputs=torch.cat([bank.base_inputs for bank in banks], dim=0),
        source_inputs=torch.cat([bank.source_inputs for bank in banks], dim=0),
        base_labels=torch.cat([bank.base_labels for bank in banks], dim=0),
        cf_labels_by_var={
            variable: torch.cat([bank.cf_labels_by_var[variable] for bank in banks], dim=0)
            for variable in first.target_vars
        },
        changed_by_var={
            variable: torch.cat([bank.changed_by_var[variable] for bank in banks], dim=0)
            for variable in first.pair_policy_vars
        },
        changed_any=torch.cat([bank.changed_any for bank in banks], dim=0),
        pair_policy=pair_policy,
        pair_policy_target=pair_policy_target,
        mixed_positive_fraction=float(mixed_positive_fraction),
        target_vars=first.target_vars,
        pair_policy_vars=first.pair_policy_vars,
        pair_pool_size=None,
        pair_stats={},
    )
    return PairBank(
        split=merged.split,
        seed=merged.seed,
        base_rows=merged.base_rows,
        source_rows=merged.source_rows,
        base_inputs=merged.base_inputs,
        source_inputs=merged.source_inputs,
        base_labels=merged.base_labels,
        cf_labels_by_var=merged.cf_labels_by_var,
        changed_by_var=merged.changed_by_var,
        changed_any=merged.changed_any,
        pair_policy=merged.pair_policy,
        pair_policy_target=merged.pair_policy_target,
        mixed_positive_fraction=merged.mixed_positive_fraction,
        target_vars=merged.target_vars,
        pair_policy_vars=merged.pair_policy_vars,
        pair_pool_size=merged.pair_pool_size,
        pair_stats=_pair_stats_from_bank(merged),
    )


def _build_shared_bank(
    problem,
    *,
    size: int,
    seed: int,
    split: str,
    target: str,
    positive_fraction: float,
    pair_pool_size: int,
) -> PairBank:
    """Build one shared bank using the standard mixed-pair constructor."""
    return build_pair_bank(
        problem,
        size,
        seed,
        split,
        target_vars=tuple(TARGET_VARS),
        pair_policy="mixed",
        pair_policy_target=target,
        mixed_positive_fraction=positive_fraction,
        pair_pool_size=pair_pool_size,
    )


def _build_calibration_bank(problem, seed: int):
    """Build calibration banks using any strategy from the calibration sweep."""
    strategy = str(CALIBRATION_STRATEGY)
    base_seed = int(seed) + 300
    if strategy in {"targeted_eval", "sensitive_test_eval"}:
        return {
            variable: _build_shared_bank(
                problem,
                size=CALIBRATION_PAIR_SIZE,
                seed=base_seed + 1 + idx,
                split=f"calibration_{variable.lower()}",
                target=variable,
                positive_fraction=SENSITIVE_TEST_MIXED_POSITIVE_FRACTION,
                pair_pool_size=SENSITIVE_TEST_PAIR_POOL_SIZE,
            )
            for idx, variable in enumerate(TARGET_VARS)
        }
    if strategy == "shared_wx_positive":
        return _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 1, split="calibration", target="WX", positive_fraction=1.0, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE)
    if strategy == "shared_yz_positive":
        return _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 2, split="calibration", target="YZ", positive_fraction=1.0, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE)
    if strategy == "shared_any_positive":
        return _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 3, split="calibration", target="any", positive_fraction=1.0, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE)
    if strategy == "shared_any_mixed50":
        return _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 4, split="calibration", target="any", positive_fraction=0.5, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE)
    if strategy == "shared_both_positive":
        return _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 5, split="calibration", target="both", positive_fraction=1.0, pair_pool_size=max(CALIBRATION_PAIR_POOL_SIZE, 8192))
    if strategy == "shared_balanced_wx_yz_only":
        wx_bank = _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE // 2, seed=base_seed + 6, split="calibration_wx_only", target="WX_only", positive_fraction=1.0, pair_pool_size=max(CALIBRATION_PAIR_POOL_SIZE, 4096))
        yz_bank = _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE // 2, seed=base_seed + 7, split="calibration_yz_only", target="YZ_only", positive_fraction=1.0, pair_pool_size=max(CALIBRATION_PAIR_POOL_SIZE, 4096))
        return _concat_pair_banks(
            [wx_bank, yz_bank],
            split="calibration",
            seed=base_seed + 8,
            pair_policy="mixed",
            pair_policy_target="balanced_wx_yz_only",
            mixed_positive_fraction=1.0,
        )
    if strategy == "separate_variable_positive":
        return {
            "WX": _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 9, split="calibration_wx", target="WX", positive_fraction=1.0, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE),
            "YZ": _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 10, split="calibration_yz", target="YZ", positive_fraction=1.0, pair_pool_size=CALIBRATION_PAIR_POOL_SIZE),
        }
    if strategy == "separate_variable_only":
        return {
            "WX": _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 11, split="calibration_wx_only", target="WX_only", positive_fraction=1.0, pair_pool_size=max(CALIBRATION_PAIR_POOL_SIZE, 4096)),
            "YZ": _build_shared_bank(problem, size=CALIBRATION_PAIR_SIZE, seed=base_seed + 12, split="calibration_yz_only", target="YZ_only", positive_fraction=1.0, pair_pool_size=max(CALIBRATION_PAIR_POOL_SIZE, 4096)),
        }
    raise ValueError(f"Unsupported CALIBRATION_STRATEGY={strategy}")


def _build_test_bank(problem, seed: int):
    """Build the evaluation holdout bank(s)."""
    return {
        variable: build_pair_bank(
            problem,
            TEST_PAIR_SIZE,
            int(seed) + 401 + idx,
            "test",
            target_vars=tuple(TARGET_VARS),
            pair_policy=SENSITIVE_TEST_PAIR_POLICY,
            pair_policy_target=variable,
            mixed_positive_fraction=SENSITIVE_TEST_MIXED_POSITIVE_FRACTION,
            pair_pool_size=SENSITIVE_TEST_PAIR_POOL_SIZE,
        )
        for idx, variable in enumerate(TARGET_VARS)
    }


def _build_invariant_test_bank(problem, seed: int):
    """Build per-variable holdout banks containing only invariant examples."""
    if not INVARIANT_TEST_EVAL:
        return None
    return {
        variable: build_pair_bank(
            problem,
            TEST_PAIR_SIZE,
            int(seed) + 501 + idx,
            "test_invariant",
            target_vars=tuple(TARGET_VARS),
            pair_policy=INVARIANT_TEST_PAIR_POLICY,
            pair_policy_target=variable,
            mixed_positive_fraction=INVARIANT_TEST_MIXED_POSITIVE_FRACTION,
            pair_pool_size=INVARIANT_TEST_PAIR_POOL_SIZE,
        )
        for idx, variable in enumerate(TARGET_VARS)
    }


def build_compare_config(
    seed: int,
    checkpoint_path: Path,
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
        seed=int(seed),
        checkpoint_path=checkpoint_path,
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
        calibration_pair_policy="mixed",
        calibration_pair_policy_target="strategy_defined",
        calibration_mixed_positive_fraction=0.0,
        calibration_pair_pool_size=CALIBRATION_PAIR_POOL_SIZE,
        test_pair_policy=SENSITIVE_TEST_PAIR_POLICY,
        test_pair_policy_target="per_variable",
        test_mixed_positive_fraction=SENSITIVE_TEST_MIXED_POSITIVE_FRACTION,
        test_pair_pool_size=SENSITIVE_TEST_PAIR_POOL_SIZE,
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


def _checkpoint_path_for_seed(seed: int) -> Path:
    """Resolve the checkpoint path for one seed."""
    return Path(f"models/equality_mlp_seed{int(seed)}.pt")


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
        invariant_by_variable = {
            str(record["variable"]): float(record.get("invariant_exact_acc", 0.0))
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
        lines.append(
            " " * (method_width + 2)
            + "  ".join(
                f"{variable}_inv={invariant_by_variable.get(variable, float('nan')):.4f}"
                for variable in variable_columns
            )
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
    best_invariant_by_variable = {
        str(record["variable"]): float(record.get("invariant_exact_acc", 0.0))
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
    lines.append(f"average_invariant_exact_acc: {float(best_summary.get('invariant_exact_acc', 0.0)):.4f}")
    for variable in TARGET_VARS:
        lines.append(f"{variable}_exact_acc: {float(best_exact_by_variable.get(variable, 0.0)):.4f}")
        lines.append(f"{variable}_invariant_exact_acc: {float(best_invariant_by_variable.get(variable, 0.0)):.4f}")
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
        invariant_by_variable = {
            str(item["variable"]): float(item.get("invariant_exact_acc", 0.0))
            for item in method_records
        }
        bits = [
            f"avg={float(summary.get('exact_acc', 0.0)):.4f}",
            f"avg_inv={float(summary.get('invariant_exact_acc', 0.0)):.4f}",
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
                f"/inv={float(invariant_by_variable.get(variable, 0.0)):.4f}"
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


def _plot_heq_joint_accuracy(seed_payload: dict[str, object], output_path: Path) -> None:
    """Render the paper-style per-seed accuracy comparison chart."""
    methods = [method for method in HEQ_METHOD_ORDER if method in seed_payload.get("best_method_runs", {})]
    if not methods:
        return

    x = np.arange(len(methods), dtype=float) * 0.72
    width = 0.12
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float) * width
    variable_palette = {
        "WX": {"sensitive": "#2f6ea6", "invariant": "#9ecae1"},
        "YZ": {"sensitive": "#c4473a", "invariant": "#f4a6a0"},
    }
    average_color = "#59a14f"

    fig, ax = plt.subplots(figsize=(9.6, 3.8), constrained_layout=True)
    for method_idx, method in enumerate(methods):
        comparison = seed_payload["best_method_runs"][method]["comparison"]
        variable_results = {
            str(record["variable"]): record
            for record in comparison["method_payloads"][method]["results"]
        }
        summary = next(
            (
                record
                for record in comparison.get("method_summary", [])
                if str(record.get("method")) == method
            ),
            {},
        )
        heights = [
            float(variable_results["WX"].get("exact_acc", 0.0)),
            float(variable_results["WX"].get("invariant_exact_acc", 0.0)),
            float(variable_results["YZ"].get("exact_acc", 0.0)),
            float(variable_results["YZ"].get("invariant_exact_acc", 0.0)),
            0.5
            * (
                float(summary.get("exact_acc", 0.0))
                + float(summary.get("invariant_exact_acc", 0.0))
            ),
        ]
        colors = [
            variable_palette["WX"]["sensitive"],
            variable_palette["WX"]["invariant"],
            variable_palette["YZ"]["sensitive"],
            variable_palette["YZ"]["invariant"],
            average_color,
        ]
        for height, offset, color in zip(heights, offsets, colors):
            ax.bar(
                x[method_idx] + offset,
                height,
                width=width,
                color=color,
                edgecolor="none",
                linewidth=0.0,
            )

    ax.set_title("HEQ: Equality Run Accuracy")
    ax.set_ylabel("Exact Accuracy")
    ax.set_xticks(x, [HEQ_METHOD_LABELS[method] for method in methods])
    ax.set_xlim(x[0] - 0.34, x[-1] + 0.34)
    ax.set_ylim(0.90, 1.01)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(
        handles=[
            Patch(facecolor=variable_palette["WX"]["sensitive"], label=r"$z_{WX}$ sensitive"),
            Patch(facecolor=variable_palette["WX"]["invariant"], label=r"$z_{WX}$ invariant"),
            Patch(facecolor=variable_palette["YZ"]["sensitive"], label=r"$z_{YZ}$ sensitive"),
            Patch(facecolor=variable_palette["YZ"]["invariant"], label=r"$z_{YZ}$ invariant"),
            Patch(facecolor=average_color, label="Average"),
        ],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.3,
    )
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_heq_joint_runtime(seed_payload: dict[str, object], output_path: Path) -> None:
    """Render the per-seed runtime comparison chart."""
    methods = [method for method in HEQ_METHOD_ORDER if method in seed_payload.get("best_method_runs", {})]
    if not methods:
        return

    runtimes = np.array(
        [
            float(seed_payload["best_method_runs"][method]["comparison"]["method_runtime_seconds"][method])
            for method in methods
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.45), constrained_layout=True)
    ax.bar(
        np.arange(len(methods)),
        runtimes,
        color=[HEQ_METHOD_COLORS[method] for method in methods],
        alpha=0.97,
    )
    ax.set_title("HEQ: Equality Run Runtime")
    ax.set_ylabel("Runtime (sec)")
    ax.set_xticks(np.arange(len(methods)), [HEQ_METHOD_LABELS[method] for method in methods])
    ax.grid(axis="y", alpha=0.22)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _make_tail_detail_norm(exponent: float = 0.68) -> FuncNorm:
    """Increase visual contrast in both tails of a unit-interval heatmap."""

    def forward(values):
        values = np.asarray(values, dtype=float)
        clipped = np.clip(values, 0.0, 1.0)
        mapped = np.empty_like(clipped, dtype=float)
        lower = clipped <= 0.5
        mapped[lower] = 0.5 * np.power(2.0 * clipped[lower], exponent)
        mapped[~lower] = 1.0 - 0.5 * np.power(2.0 * (1.0 - clipped[~lower]), exponent)
        return mapped

    def inverse(values):
        values = np.asarray(values, dtype=float)
        clipped = np.clip(values, 0.0, 1.0)
        mapped = np.empty_like(clipped, dtype=float)
        lower = clipped <= 0.5
        mapped[lower] = 0.5 * np.power(2.0 * clipped[lower], 1.0 / exponent)
        mapped[~lower] = 1.0 - 0.5 * np.power(2.0 * (1.0 - clipped[~lower]), 1.0 / exponent)
        return mapped

    return FuncNorm((forward, inverse), vmin=0.0, vmax=1.0)


def _selected_das_projector_diags(seed_payload: dict[str, object]) -> dict[str, np.ndarray]:
    """Recover the selected DAS projector diagonals from saved rotation weights."""
    if "das" not in seed_payload.get("best_method_runs", {}):
        return {}

    records = seed_payload["best_method_runs"]["das"]["comparison"]["method_payloads"]["das"]["results"]
    projector_by_variable: dict[str, np.ndarray] = {}
    for record in records:
        rotations = list(dict(record.get("selected_rotation", {})).get("rotations", []))
        if not rotations:
            continue
        weight = np.asarray(rotations[0].get("weight", []), dtype=float)
        if weight.ndim != 2 or weight.size == 0:
            continue
        subspace_dim = int(record.get("subspace_dim", 0))
        if subspace_dim <= 0 or subspace_dim > weight.shape[1]:
            continue
        projector_diag = np.square(weight[:, :subspace_dim]).sum(axis=1)
        total = float(projector_diag.sum())
        if total > 0.0:
            projector_diag = projector_diag / total
        projector_by_variable[str(record["variable"])] = projector_diag
    return projector_by_variable


def _plot_heq_handle_summary(seed_payload: dict[str, object], output_path: Path) -> None:
    """Render the paper-style handle summary and crop left/right panels."""
    methods = [method for method in HEQ_METHOD_ORDER if method in seed_payload.get("best_method_runs", {})]
    if not methods:
        return

    variables = ["WX", "YZ"]
    layer_labels = ["L1", "L2", "L3"]
    variable_colors = {"WX": "#4e79a7", "YZ": "#e15759"}
    layer_mass = {
        variable: {method: np.zeros(len(layer_labels), dtype=float) for method in methods}
        for variable in variables
    }
    transport_methods = [method for method in methods if method in {"ot", "uot"}]
    transport_candidate_heatmaps = {
        method: {variable: np.zeros((3, 16), dtype=float) for variable in variables}
        for method in transport_methods
    }
    transport_topk_heatmaps = {
        method: {variable: np.zeros((3, 16), dtype=float) for variable in variables}
        for method in transport_methods
    }
    das_canonical_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    das_rotated_heatmaps = {variable: np.zeros((3, 16), dtype=float) for variable in variables}
    das_projector_diags = _selected_das_projector_diags(seed_payload)

    for method in methods:
        comparison = seed_payload["best_method_runs"][method]["comparison"]
        method_payload = comparison["method_payloads"][method]
        method_selection = comparison["method_selections"][method]
        if method in {"ot", "uot"}:
            transport_matrix = np.asarray(method_payload["transport"], dtype=float)
            target_vars = [str(variable) for variable in method_payload["target_vars"]]
            site_labels = [str(label) for label in method_payload["sites"]]
            for target_idx, variable in enumerate(target_vars):
                candidate_values = transport_matrix[target_idx]
                total_mass = float(candidate_values.sum())
                for site_idx, site_label in enumerate(site_labels):
                    layer_idx = int(site_label[1])
                    dim_idx = int(site_label.split("-d", 1)[1])
                    if 0 <= layer_idx < 3 and 0 <= dim_idx < 16:
                        value = float(candidate_values[site_idx])
                        transport_candidate_heatmaps[method][variable][layer_idx, dim_idx] = value
                        if total_mass > 0.0:
                            layer_mass[variable][method][layer_idx] += value / total_mass
        for record in method_selection["results"]:
            variable = str(record["variable"])
            if method == "das":
                layer_index = int(record["layer"])
                layer_mass[variable][method][layer_index] = 1.0
                projector_diag = np.asarray(das_projector_diags.get(variable, np.zeros(16, dtype=float)), dtype=float)
                width = min(projector_diag.shape[0], das_canonical_heatmaps[variable].shape[1])
                das_canonical_heatmaps[variable][layer_index, :width] = projector_diag[:width]
                das_rotated_heatmaps[variable][layer_index, : int(record["subspace_dim"])] = 1.0
            else:
                for layer_name, mask_values in method_payload["layer_masks_by_variable"][variable].items():
                    layer_idx = int(layer_name[1:])
                    transport_topk_heatmaps[method][variable][layer_idx, :] = np.asarray(mask_values, dtype=float)

    fig = plt.figure(figsize=(18.3, 3.35), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 3.58], wspace=0.22)
    left_grid = outer[0, 0].subgridspec(3, 1, height_ratios=[0.079, 0.92, 0.001], hspace=0.0)
    ax = fig.add_subplot(left_grid[1, 0])
    right_outer = outer[0, 1].subgridspec(
        len(methods) + 1,
        5,
        height_ratios=[0.11] + [1.0] * len(methods),
        width_ratios=[3.55, 0.21, 3.55, 0.08, 0.11],
        wspace=0.006,
        hspace=0.0,
    )

    ordered_pairs = [(method, variable) for method in methods for variable in variables]
    x = np.arange(len(ordered_pairs))
    layer_bar_width = 0.18
    min_visible_bar_height = 0.006
    layer_offsets = np.array([-layer_bar_width, 0.0, layer_bar_width], dtype=float)
    for layer_idx in range(len(layer_labels)):
        heights = np.array(
            [layer_mass[variable][method][layer_idx] for method, variable in ordered_pairs],
            dtype=float,
        )
        display_heights = np.where(heights > 0.0, heights, min_visible_bar_height)
        facecolors = [variable_colors[variable] for _, variable in ordered_pairs]
        ax.bar(
            x + layer_offsets[layer_idx],
            display_heights,
            width=layer_bar_width,
            color=facecolors,
            alpha=0.35 + 0.28 * layer_idx,
            edgecolor=facecolors,
            linewidth=0.7,
        )
    ax.set_xticks(
        x,
        [rf"{HEQ_METHOD_LABELS[method]}" + "\n" + rf"$z_{{{variable}}}$" for method, variable in ordered_pairs],
    )
    ax.set_ylim(0.0, 1.03)
    ax.set_ylabel("Normalized layer mass")
    ax.set_title("Layer weight distribution pre-calibration")
    ax.grid(True, axis="y", alpha=0.2)
    for boundary in [index + 1.5 for index in range(len(methods) - 1)]:
        ax.axvline(boundary, color="#999999", linewidth=0.8, alpha=0.5)

    label_x_offsets_by_pair = {
        0: np.array([-0.62 * layer_bar_width, 0.0, 0.5 * layer_bar_width], dtype=float),
        1: np.array([-0.72 * layer_bar_width, -0.22 * layer_bar_width, 0.0], dtype=float),
    }
    label_gap = 0.024
    for pair_idx in range(min(2, len(ordered_pairs))):
        group_color = variable_colors[ordered_pairs[pair_idx][1]]
        label_x_offsets = label_x_offsets_by_pair[pair_idx]
        for layer_idx, layer_text in enumerate(layer_labels):
            base_height = float(layer_mass[ordered_pairs[pair_idx][1]][ordered_pairs[pair_idx][0]][layer_idx])
            ax.text(
                x[pair_idx] + layer_offsets[layer_idx] + label_x_offsets[layer_idx],
                base_height + label_gap,
                layer_text,
                color=group_color,
                fontsize=8.5,
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    cmap = plt.get_cmap("viridis")
    norm = _make_tail_detail_norm()
    for variable, grid_col in zip(variables, [0, 2]):
        title_ax = fig.add_subplot(right_outer[0, grid_col])
        title_ax.axis("off")
        title_ax.text(0.5, 0.76, rf"$z_{{{variable}}}$", ha="center", va="center", fontsize=12)

    image = None
    variable_blocks = {
        "WX": right_outer[1:, 0].subgridspec(len(methods), 2, wspace=0.01, hspace=0.0),
        "YZ": right_outer[1:, 2].subgridspec(len(methods), 2, wspace=0.01, hspace=0.0),
    }
    for row_idx, method in enumerate(methods):
        for variable in variables:
            pair_grid = variable_blocks[variable]
            first_ax = fig.add_subplot(pair_grid[row_idx, 0])
            second_ax = fig.add_subplot(pair_grid[row_idx, 1], sharey=first_ax)
            if method in {"ot", "uot"}:
                first_values = np.asarray(transport_candidate_heatmaps[method][variable], dtype=float)
                second_values = np.asarray(transport_topk_heatmaps[method][variable], dtype=float)
                first_title = "pre-top$K$"
                second_title = "top-$K$"
            else:
                first_values = np.asarray(das_canonical_heatmaps[variable], dtype=float)
                second_values = np.asarray(das_rotated_heatmaps[variable], dtype=float)
                first_title = "canonical"
                second_title = "rotated"

            def rescale_panel(values: np.ndarray) -> np.ndarray:
                values = np.asarray(values, dtype=float)
                vmax = float(values.max())
                vmin = float(values.min())
                if vmax <= 0.0:
                    return values
                if vmax > vmin:
                    return (values - vmin) / (vmax - vmin)
                return values / vmax

            first_values = rescale_panel(first_values)
            second_values = rescale_panel(second_values)
            box_aspect = first_values.shape[0] / first_values.shape[1]
            first_ax.set_box_aspect(box_aspect)
            second_ax.set_box_aspect(box_aspect)
            image = first_ax.imshow(first_values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
            second_ax.imshow(second_values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

            for heat_ax in (first_ax, second_ax):
                heat_ax.set_xticks(np.arange(-0.5, 16, 1), minor=True)
                heat_ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
                heat_ax.grid(which="minor", color="white", linewidth=0.32)
                heat_ax.tick_params(which="minor", bottom=False, left=False)
                for spine in heat_ax.spines.values():
                    spine.set_linewidth(0.9)
                    spine.set_edgecolor("#333333")

            if variable == "WX":
                first_ax.set_yticks([0, 1, 2], layer_labels)
                first_ax.set_ylabel(HEQ_METHOD_LABELS[method], rotation=0, labelpad=12, va="center", fontsize=11)
            else:
                first_ax.set_yticks([0, 1, 2], [])
            second_ax.set_yticks([0, 1, 2], [])

            if row_idx == len(methods) - 1:
                first_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
                second_ax.set_xticks([0, 5, 10, 15], ["0", "5", "10", "15"])
                first_ax.set_xlabel("Neuron index")
            else:
                first_ax.set_xticks([0, 5, 10, 15], [])
                second_ax.set_xticks([0, 5, 10, 15], [])

            first_ax.set_title(first_title, fontsize=8.5, pad=2.0, color="#444444")
            second_ax.set_title(second_title, fontsize=8.5, pad=2.0, color="#444444")

    if image is not None:
        cax = fig.add_subplot(right_outer[1:, 4])
        tick_values = [0.0, 0.05, 0.15, 0.5, 0.85, 0.95, 1.0]
        cbar = fig.colorbar(image, cax=cax, ticks=tick_values)
        cbar.ax.set_yticklabels([f"{tick:.2f}".rstrip("0").rstrip(".") for tick in tick_values])
        cbar.ax.tick_params(labelsize=8.5)
        cbar.set_label("Relative site strength", rotation=90)

    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    image_file = Image.open(output_path)
    width, height = image_file.size
    left_end = int(round(0.229 * width))
    right_start = int(round(0.244 * width))
    image_file.crop((0, 0, left_end, height)).save(output_path.with_name("heq_handle_summary_left.png"))
    image_file.crop((right_start, 0, width, height)).save(output_path.with_name("heq_handle_summary_right.png"))


def _save_heq_seed_plots(seed_payload: dict[str, object], seed_run_dir: Path) -> dict[str, str]:
    """Generate the HEQ summary plots emitted at the end of one seed run."""
    plot_paths: dict[str, str] = {}

    accuracy_path = seed_run_dir / "heq_joint_accuracy.png"
    _plot_heq_joint_accuracy(seed_payload, accuracy_path)
    if accuracy_path.exists():
        plot_paths["heq_joint_accuracy"] = str(accuracy_path)

    runtime_path = seed_run_dir / "heq_joint_runtime.png"
    _plot_heq_joint_runtime(seed_payload, runtime_path)
    if runtime_path.exists():
        plot_paths["heq_joint_runtime"] = str(runtime_path)

    handle_summary_path = seed_run_dir / "heq_handle_summary.png"
    _plot_heq_handle_summary(seed_payload, handle_summary_path)
    if handle_summary_path.exists():
        plot_paths["heq_handle_summary"] = str(handle_summary_path)
    for key in ("heq_handle_summary_left", "heq_handle_summary_right"):
        path = seed_run_dir / f"{key}.png"
        if path.exists():
            plot_paths[key] = str(path)

    return plot_paths


def _build_aggregate_seed_summary(seed_payloads: list[dict[str, object]]) -> tuple[dict[str, object], str]:
    """Aggregate best-per-method exact accuracies across seeds."""
    method_to_seed_rows: dict[str, list[dict[str, object]]] = {}
    for seed_payload in seed_payloads:
        seed = int(seed_payload["seed"])
        best_method_runs = dict(seed_payload.get("best_method_runs", {}))
        for method, record in best_method_runs.items():
            comparison = dict(record.get("comparison", {}))
            method_records = [
                dict(item)
                for item in comparison.get("results", [])
                if str(item.get("method")) == method
            ]
            exact_by_variable = {
                str(item["variable"]): float(item.get("exact_acc", 0.0))
                for item in method_records
            }
            invariant_by_variable = {
                str(item["variable"]): float(item.get("invariant_exact_acc", 0.0))
                for item in method_records
            }
            avg = float(
                next(
                    (
                        item.get("exact_acc", 0.0)
                        for item in comparison.get("method_summary", [])
                        if str(item.get("method")) == method
                    ),
                    0.0,
                )
            )
            invariant_avg = float(
                next(
                    (
                        item.get("invariant_exact_acc", 0.0)
                        for item in comparison.get("method_summary", [])
                        if str(item.get("method")) == method
                    ),
                    0.0,
                )
            )
            method_to_seed_rows.setdefault(method, []).append(
                {
                    "seed": seed,
                    "average_exact_acc": avg,
                    "average_invariant_exact_acc": invariant_avg,
                    "exact_by_variable": exact_by_variable,
                    "invariant_by_variable": invariant_by_variable,
                    "source": dict(record),
                }
            )

    aggregate_rows = []
    lines = ["Aggregate Best-Per-Method Accuracy Across Seeds", f"seeds: {', '.join(str(int(seed_payload['seed'])) for seed_payload in seed_payloads)}", ""]
    for method in sorted(method_to_seed_rows):
        rows = method_to_seed_rows[method]
        avg_values = np.asarray([float(row["average_exact_acc"]) for row in rows], dtype=float)
        invariant_avg_values = np.asarray([float(row["average_invariant_exact_acc"]) for row in rows], dtype=float)
        variable_stats = {}
        for variable in TARGET_VARS:
            values = np.asarray([float(row["exact_by_variable"].get(variable, 0.0)) for row in rows], dtype=float)
            invariant_values = np.asarray([float(row["invariant_by_variable"].get(variable, 0.0)) for row in rows], dtype=float)
            variable_stats[variable] = {
                "mean": float(values.mean()) if values.size else 0.0,
                "std": float(values.std()) if values.size else 0.0,
                "values": [float(value) for value in values.tolist()],
                "invariant_mean": float(invariant_values.mean()) if invariant_values.size else 0.0,
                "invariant_std": float(invariant_values.std()) if invariant_values.size else 0.0,
                "invariant_values": [float(value) for value in invariant_values.tolist()],
            }
        aggregate_rows.append(
            {
                "method": method,
                "average_exact_acc_mean": float(avg_values.mean()) if avg_values.size else 0.0,
                "average_exact_acc_std": float(avg_values.std()) if avg_values.size else 0.0,
                "average_invariant_exact_acc_mean": float(invariant_avg_values.mean()) if invariant_avg_values.size else 0.0,
                "average_invariant_exact_acc_std": float(invariant_avg_values.std()) if invariant_avg_values.size else 0.0,
                "variables": variable_stats,
                "per_seed": rows,
            }
        )
        lines.append(f"{method.upper()}")
        for variable in TARGET_VARS:
            stats = variable_stats[variable]
            lines.append(
                f"{variable} = {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(per-seed: {', '.join(f'{value:.4f}' for value in stats['values'])})"
            )
            lines.append(
                f"{variable}_invariant = {stats['invariant_mean']:.4f} ± {stats['invariant_std']:.4f} "
                f"(per-seed: {', '.join(f'{value:.4f}' for value in stats['invariant_values'])})"
            )
        lines.append(
            f"average = {float(avg_values.mean()):.4f} ± {float(avg_values.std()):.4f} "
            f"(per-seed: {', '.join(f'{float(value):.4f}' for value in avg_values.tolist())})"
        )
        lines.append(
            f"average_invariant = {float(invariant_avg_values.mean()):.4f} ± {float(invariant_avg_values.std()):.4f} "
            f"(per-seed: {', '.join(f'{float(value):.4f}' for value in invariant_avg_values.tolist())})"
        )
        lines.append("")
    return {"methods": aggregate_rows}, "\n".join(lines).rstrip()


def _run_single_seed(seed: int) -> dict[str, object]:
    """Run the full HEQ comparison pipeline for one seed."""
    seed_run_dir = RUN_DIR / f"seed_{int(seed)}"
    checkpoint_path = _checkpoint_path_for_seed(seed)
    output_path = seed_run_dir / "equality_run_results.json"
    summary_path = seed_run_dir / "equality_run_summary.txt"

    problem = load_equality_problem(
        run_checks=True,
        num_entities=NUM_ENTITIES,
        embedding_dim=EMBEDDING_DIM,
        seed=int(seed),
    )
    device = resolve_device(DEVICE)
    train_config = build_train_config_for_seed(seed)

    if RETRAIN_BACKBONE or not checkpoint_path.exists():
        model, _, backbone_meta = train_backbone(
            problem=problem,
            train_config=train_config,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    else:
        model, _, backbone_meta = load_backbone(
            problem=problem,
            checkpoint_path=checkpoint_path,
            device=device,
            train_config=train_config,
        )

    train_bank = build_pair_bank(
        problem,
        TRAIN_PAIR_SIZE,
        int(seed) + 201,
        "train",
        target_vars=tuple(TARGET_VARS),
        pair_policy=TRAIN_PAIR_POLICY,
        pair_policy_target=TRAIN_PAIR_POLICY_TARGET,
        mixed_positive_fraction=TRAIN_MIXED_POSITIVE_FRACTION,
        pair_pool_size=TRAIN_PAIR_POOL_SIZE,
    )
    calibration_bank = _build_calibration_bank(problem, int(seed))
    test_bank = _build_test_bank(problem, int(seed))
    invariant_test_bank = _build_invariant_test_bank(problem, int(seed))
    transport_prepare_cache: dict[tuple[object, ...], dict[str, object]] = {}

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
        f"seed: {int(seed)}",
        f"device: {device}",
        f"methods: {', '.join(METHODS)}",
        f"target_vars: {', '.join(TARGET_VARS)}",
        f"num_entities: {NUM_ENTITIES}",
        f"embedding_dim: {EMBEDDING_DIM}",
        f"checkpoint_path: {checkpoint_path}",
        (
            "pair_sizes: "
            f"train={TRAIN_PAIR_SIZE}, "
            f"calibration={CALIBRATION_PAIR_SIZE}, "
            f"test={TEST_PAIR_SIZE}"
        ),
        f"calibration_strategy: {CALIBRATION_STRATEGY}",
        f"sensitive_test_eval: {SENSITIVE_TEST_EVAL}",
        f"train_pair_policy: {TRAIN_PAIR_POLICY}",
        f"train_pair_policy_target: {TRAIN_PAIR_POLICY_TARGET}",
        f"train_mixed_positive_fraction: {TRAIN_MIXED_POSITIVE_FRACTION}",
        f"train_pair_pool_size: {TRAIN_PAIR_POOL_SIZE}",
        f"calibration_pair_pool_size: {CALIBRATION_PAIR_POOL_SIZE}",
        f"test_pair_policy: {SENSITIVE_TEST_PAIR_POLICY}",
        "test_pair_policy_target: per-variable sensitive pairs",
        f"test_mixed_positive_fraction: {SENSITIVE_TEST_MIXED_POSITIVE_FRACTION}",
        f"test_pair_pool_size: {SENSITIVE_TEST_PAIR_POOL_SIZE}",
        f"invariant_test_eval: {INVARIANT_TEST_EVAL}",
        f"invariant_test_pair_policy: {INVARIANT_TEST_PAIR_POLICY}",
        "invariant_test_pair_policy_target: per-variable invariant pairs",
        f"invariant_test_mixed_positive_fraction: {INVARIANT_TEST_MIXED_POSITIVE_FRACTION}",
        f"invariant_test_pair_pool_size: {INVARIANT_TEST_PAIR_POOL_SIZE}",
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
        "",
    ]
    main_summary_lines.extend(_pair_bank_summary_lines(train_bank))
    main_summary_lines.extend(_pair_bank_summary_lines(calibration_bank))
    main_summary_lines.extend(_pair_bank_summary_lines(test_bank))
    if invariant_test_bank is not None:
        main_summary_lines.extend(_pair_bank_summary_lines(invariant_test_bank))
    main_summary_lines.append("")

    static_runs = []
    if NON_TRANSPORT_METHODS:
        for method in NON_TRANSPORT_METHODS:
            print(f"[seed {seed}] [fixed] method={method}")
            method_dir = seed_run_dir / method
            comparison = run_comparison_with_banks(
                model=model,
                backbone_meta=backbone_meta,
                device=device,
                config=build_compare_config(
                    int(seed),
                    checkpoint_path,
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
                invariant_test_bank=invariant_test_bank,
                transport_prepare_cache=transport_prepare_cache,
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
            method_run_dir = seed_run_dir / method
            config_stem = _build_transport_config_stem(
                method=method,
                signature_mode=signature_mode,
                ot_epsilon=ot_epsilon,
                ot_tau=ot_tau,
                uot_beta_abstract=uot_beta_abstract,
                uot_beta_neural=uot_beta_neural,
            )
            print(
                f"[seed {seed}] [sweep {sweep_index}/{len(method_sweep_points)}] "
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
                    int(seed),
                    checkpoint_path,
                    (method,),
                    ot_epsilon,
                    ot_tau,
                    uot_beta_abstract,
                    uot_beta_neural,
                    signature_mode,
                    method_run_dir / f"{config_stem}_results.json",
                    method_run_dir / f"{config_stem}_summary.txt",
                ),
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                test_bank=test_bank,
                invariant_test_bank=invariant_test_bank,
                transport_prepare_cache=transport_prepare_cache,
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
        method_dir = seed_run_dir / method
        result_path = method_dir / f"{method}_results.json"
        method_summary_path = method_dir / f"{method}_summary.txt"
        comparison["summary_path"] = str(method_summary_path)
        write_json(result_path, comparison)
        summary_text = f"{method.upper()} Summary"
        if "method_selections" in comparison:
            summary_entry = dict(comparison["method_selections"].get(method, {}))
            if summary_entry:
                summary_text = format_method_selection_summary(summary_entry)
        write_text_report(method_summary_path, summary_text)
        best_method_runs[method] = {
            "method": method,
            "source_type": "fixed",
            "output_path": str(result_path),
            "summary_path": str(method_summary_path),
            "comparison": comparison,
        }

    payload = {
        "seed": int(seed),
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "retrain_backbone": RETRAIN_BACKBONE,
        "calibration_strategy": CALIBRATION_STRATEGY,
        "sensitive_test_eval": SENSITIVE_TEST_EVAL,
        "ot_epsilons": [float(value) for value in OT_EPSILONS],
        "ot_taus": [float(value) for value in OT_TAUS],
        "uot_beta_abstracts": [float(value) for value in UOT_BETA_ABSTRACTS],
        "uot_beta_neurals": [float(value) for value in UOT_BETA_NEURALS],
        "signature_modes": list(SIGNATURE_MODES),
        "target_vars": list(TARGET_VARS),
        "best_method_runs": {},
        "summary_path": str(summary_path),
    }
    for method in TRANSPORT_METHODS:
        method_records = [record for record in epsilon_sweeps if str(record.get("method")) == method]
        if not method_records:
            continue
        best_record = _best_record_for_method(method, method_records)
        if best_record is None:
            continue
        method_dir = seed_run_dir / method
        result_path = method_dir / f"{method}_results.json"
        method_summary_path = method_dir / f"{method}_summary.txt"
        comparison = dict(best_record.get("comparison", {}))
        comparison["summary_path"] = str(method_summary_path)
        write_json(result_path, comparison)
        write_text_report(method_summary_path, _build_transport_method_summary(method, method_records))
        best_method_runs[method] = {
            "method": method,
            "source_type": "sweep",
            "ot_epsilon": float(best_record.get("ot_epsilon", 0.0)),
            "ot_tau": float(best_record.get("ot_tau", 0.0)),
            "uot_beta_abstract": float(best_record.get("uot_beta_abstract", 0.0)),
            "uot_beta_neural": float(best_record.get("uot_beta_neural", 0.0)),
            "signature_mode": best_record.get("signature_mode"),
            "output_path": str(result_path),
            "summary_path": str(method_summary_path),
            "comparison": comparison,
        }
    payload["best_method_runs"] = best_method_runs
    best_method_table_records, best_method_table_text = _build_best_method_exact_table(
        list(best_method_runs.values()),
        [],
    )
    best_method_plot_path = _save_best_method_exact_plot(
        best_method_table_records,
        seed_run_dir / "best_method_exact_accuracy.png",
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

    payload["heq_plots"] = _save_heq_seed_plots(payload, seed_run_dir)
    write_json(output_path, payload)
    write_text_report(
        summary_path,
        "\n".join(main_summary_lines)
        + "\n\n"
        + best_method_table_text
        + "\n\n"
        + (f"best_method_exact_plot: {best_method_plot_path}\n\n" if best_method_plot_path is not None else "")
        + ("\n\n" + ("=" * 72) + "\n\n").join(best_method_sections),
    )
    return payload


def main() -> None:
    seed_payloads = []
    for seed_index, seed in enumerate(SEEDS, start=1):
        print(f"[seed {seed_index}/{len(SEEDS)}] seed={int(seed)}")
        seed_payloads.append(_run_single_seed(int(seed)))

    aggregate_payload, aggregate_text = _build_aggregate_seed_summary(seed_payloads)
    aggregate_rows = []
    for item in aggregate_payload["methods"]:
        aggregate_rows.append(
            {
                "method": item["method"],
                "average_exact_acc": float(item["average_exact_acc_mean"]),
                "records": [
                    {
                        "variable": variable,
                        "exact_acc": float(stats["mean"]),
                    }
                    for variable, stats in item["variables"].items()
                ],
                "source": {"source_type": "aggregate"},
            }
        )
    best_method_plot_path = _save_best_method_exact_plot(
        aggregate_rows,
        RUN_DIR / "best_method_exact_accuracy.png",
    )
    payload = {
        "seeds": [int(seed) for seed in SEEDS],
        "device": str(resolve_device(DEVICE)),
        "retrain_backbone": RETRAIN_BACKBONE,
        "calibration_strategy": CALIBRATION_STRATEGY,
        "sensitive_test_eval": SENSITIVE_TEST_EVAL,
        "ot_epsilons": [float(value) for value in OT_EPSILONS],
        "ot_taus": [float(value) for value in OT_TAUS],
        "uot_beta_abstracts": [float(value) for value in UOT_BETA_ABSTRACTS],
        "uot_beta_neurals": [float(value) for value in UOT_BETA_NEURALS],
        "signature_modes": list(SIGNATURE_MODES),
        "target_vars": list(TARGET_VARS),
        "seed_runs": seed_payloads,
        "aggregate": aggregate_payload,
        "best_method_exact_plot": best_method_plot_path,
        "summary_path": str(SUMMARY_PATH),
    }
    lines = [
        "Hierarchical Equality Run Summary",
        f"seeds: {', '.join(str(int(seed)) for seed in SEEDS)}",
        f"methods: {', '.join(METHODS)}",
        f"target_vars: {', '.join(TARGET_VARS)}",
        f"transport_solver_backend: {TRANSPORT_SOLVER_BACKEND}",
        "",
        aggregate_text,
    ]
    if best_method_plot_path is not None:
        lines.extend(["", f"best_method_exact_plot: {best_method_plot_path}"])
    write_json(OUTPUT_PATH, payload)
    write_text_report(SUMMARY_PATH, "\n".join(lines))

    print(aggregate_text)
    if best_method_plot_path is not None:
        print(f"Wrote best-method exact plot to {Path(best_method_plot_path).resolve()}")
    print(f"Wrote run results to {Path(OUTPUT_PATH).resolve()}")
    print(f"Wrote run summary to {Path(SUMMARY_PATH).resolve()}")


if __name__ == "__main__":
    main()
