"""Reusable comparison runner for single-seed and multi-seed experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .backbone import AdditionTrainConfig, load_backbone
from .constants import (
    DEFAULT_ALIGNMENT_RESOLUTION,
    DEFAULT_FACTUAL_VALIDATION_SIZE,
    DEFAULT_PAIR_TEST_SIZE,
    DEFAULT_PAIR_TRAIN_SIZE,
    DEFAULT_TARGET_VARS,
)
from .das import DASConfig, run_das_pipeline
from .ot import OTConfig, run_alignment_pipeline
from .pair_bank import build_pair_bank
from .plots import save_comparison_plots
from .reporting import (
    build_method_selection_summary,
    format_method_candidate_sweep,
    format_method_selection_summary,
    print_results_table,
    summarize_method_records,
    write_text_report,
)
from .runtime import write_json


@dataclass(frozen=True)
class CompareExperimentConfig:
    """Config controlling one end-to-end comparison run for a fixed seed."""

    seed: int
    checkpoint_path: Path
    output_path: Path
    summary_path: Path
    methods: tuple[str, ...] = ("gw", "ot", "fgw", "das")
    factual_validation_size: int = DEFAULT_FACTUAL_VALIDATION_SIZE
    train_pair_size: int = DEFAULT_PAIR_TRAIN_SIZE
    calibration_pair_size: int = DEFAULT_PAIR_TRAIN_SIZE
    test_pair_size: int = DEFAULT_PAIR_TEST_SIZE
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS
    train_pair_policy: str = "unfiltered"
    train_pair_policy_target: str = "any"
    train_mixed_positive_fraction: float = 0.5
    train_pair_pool_size: int | None = None
    calibration_pair_policy: str = "unfiltered"
    calibration_pair_policy_target: str = "any"
    calibration_mixed_positive_fraction: float = 0.5
    calibration_pair_pool_size: int | None = None
    test_pair_policy: str = "unfiltered"
    test_pair_policy_target: str = "any"
    test_mixed_positive_fraction: float = 0.5
    test_pair_pool_size: int | None = None
    batch_size: int = 128
    resolution: int = DEFAULT_ALIGNMENT_RESOLUTION
    fgw_alpha: float = 0.5
    ot_epsilon: float = 5e-2
    ot_tau: float = 1.0
    ot_top_k_values: tuple[int, ...] | None = None
    ot_lambdas: tuple[float, ...] = (1.0,)
    das_max_epochs: int = 1
    das_min_epochs: int = 1
    das_plateau_patience: int = 2
    das_plateau_rel_delta: float = 5e-3
    das_learning_rate: float = 1e-3
    das_subspace_dims: tuple[int, ...] | None = None
    das_layers: tuple[int, ...] | None = None


def _build_summary_lines(
    config: CompareExperimentConfig,
    device,
    backbone_meta: dict[str, object],
    train_bank,
    calibration_bank,
    test_bank,
    method_payloads: dict[str, dict[str, object]],
    method_runtime_seconds: dict[str, float],
    summary_records: list[dict[str, object]],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    """Build the plain-text report lines and method selection payload."""
    method_selections = {
        method: build_method_selection_summary(method, method_payloads[method])
        for method in config.methods
    }

    factual_metrics = dict(backbone_meta.get("factual_validation_metrics", {}))
    summary_lines = [
        "Addition Compare Summary",
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
        f"ot_epsilon: {float(config.ot_epsilon):.6f}",
        f"ot_tau: {float(config.ot_tau):.6f}",
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


def run_comparison_with_model(
    *,
    problem,
    model,
    backbone_meta: dict[str, object],
    device,
    config: CompareExperimentConfig,
) -> dict[str, object]:
    """Run the shared alignment evaluation starting from an in-memory model."""
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
        print(f"[{method_index}/{len(config.methods)}] Starting {method.upper()}")
        method_start_time = perf_counter()
        if method in {"gw", "ot", "fgw"}:
            payload = run_alignment_pipeline(
                model=model,
                fit_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device=device,
                config=OTConfig(
                    method=method,
                    batch_size=config.batch_size,
                    resolution=config.resolution,
                    alpha=config.fgw_alpha,
                    epsilon=config.ot_epsilon,
                    tau=config.ot_tau,
                    target_vars=tuple(config.target_vars),
                    top_k_values=config.ot_top_k_values,
                    lambda_values=config.ot_lambdas,
                ),
            )
        elif method == "das":
            payload = run_das_pipeline(
                model=model,
                train_bank=train_bank,
                calibration_bank=calibration_bank,
                holdout_bank=test_bank,
                device=device,
                config=DASConfig(
                    batch_size=config.batch_size,
                    max_epochs=config.das_max_epochs,
                    min_epochs=config.das_min_epochs,
                    plateau_patience=config.das_plateau_patience,
                    plateau_rel_delta=config.das_plateau_rel_delta,
                    learning_rate=config.das_learning_rate,
                    subspace_dims=config.das_subspace_dims,
                    search_layers=config.das_layers,
                    target_vars=tuple(config.target_vars),
                ),
            )
        else:
            raise ValueError(f"Unsupported method {method}")
        runtime_seconds = perf_counter() - method_start_time
        payload["runtime_seconds"] = float(runtime_seconds)
        method_payloads[method] = payload
        method_runtime_seconds[method] = float(runtime_seconds)
        all_records.extend(payload["results"])
        print(f"{method.upper()} runtime: {float(runtime_seconds):.2f}s")
        print()

    summary_records = summarize_method_records(all_records)
    for record in summary_records:
        record["runtime_seconds"] = float(method_runtime_seconds.get(str(record["method"]), 0.0))
    summary_lines, method_selections = _build_summary_lines(
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
        "ot_epsilon": float(config.ot_epsilon),
        "ot_tau": float(config.ot_tau),
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
    print(f"Wrote comparison results to {Path(config.output_path).resolve()}")
    print(f"Wrote comparison summary to {Path(config.summary_path).resolve()}")
    return payload


def run_comparison_from_checkpoint(
    *,
    problem,
    device,
    backbone_train_config: AdditionTrainConfig,
    config: CompareExperimentConfig,
) -> dict[str, object]:
    """Load a compatible checkpoint and run the comparison pipeline."""
    model, _, backbone_meta = load_backbone(
        problem=problem,
        checkpoint_path=config.checkpoint_path,
        device=device,
        train_config=backbone_train_config,
    )
    return run_comparison_with_model(
        problem=problem,
        model=model,
        backbone_meta=backbone_meta,
        device=device,
        config=config,
    )
