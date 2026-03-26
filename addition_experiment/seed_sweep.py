"""Aggregation and plotting helpers for multi-seed experiment sweeps."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from . import _env  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from .runtime import ensure_parent_dir

METHOD_PASTEL_COLORS = {
    "das": "#a8d5ba",
    "fgw": "#f6bd60",
    "gw": "#9ecae1",
    "ot": "#cdb4db",
}

DEFAULT_PASTEL_COLORS = (
    "#9ecae1",
    "#f6bd60",
    "#a8d5ba",
    "#cdb4db",
    "#f7cad0",
    "#bde0fe",
)


def _shade_color(color: str, shade_index: int, num_shades: int) -> tuple[float, float, float]:
    """Return a soft shade of the base color, lighter for earlier groups."""
    base = np.asarray(to_rgb(color), dtype=float)
    if num_shades <= 1:
        mix = 0.0
    else:
        mix = 0.32 - 0.22 * (shade_index / float(num_shades - 1))
    return tuple((1.0 - mix) * base + mix * np.ones(3, dtype=float))


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Return population mean/std, handling empty and singleton lists safely."""
    if not values:
        return 0.0, 0.0
    values_np = np.asarray(values, dtype=float)
    return float(values_np.mean()), float(values_np.std(ddof=0))


def build_seed_sweep_payload(seed_runs: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-seed comparison payloads into cross-seed summaries."""
    method_average_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    variable_grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    backbone_factual = []
    per_seed_method_summary = []
    per_seed_variable_results = []
    per_seed_method_runtime = []
    methods_seen = set()
    target_vars = []
    seeds = []

    for seed_run in seed_runs:
        seed = int(seed_run["seed"])
        seeds.append(seed)
        comparison = dict(seed_run["comparison"])
        target_vars = [str(variable) for variable in comparison.get("target_vars", target_vars)]
        method_runtime_seconds = {
            str(method): float(seconds)
            for method, seconds in dict(comparison.get("method_runtime_seconds", {})).items()
        }

        backbone = dict(comparison.get("backbone", {}))
        factual_metrics = dict(backbone.get("factual_validation_metrics", {}))
        backbone_factual.append(
            {
                "seed": seed,
                "exact_acc": float(factual_metrics.get("exact_acc", 0.0)),
                "num_examples": int(factual_metrics.get("num_examples", 0)),
            }
        )

        for record in comparison.get("method_summary", []):
            method = str(record["method"])
            methods_seen.add(method)
            average_record = {
                "seed": seed,
                "method": method,
                "exact_acc": float(record["exact_acc"]),
                "mean_shared_digits": float(record["mean_shared_digits"]),
                "runtime_seconds": float(
                    record.get("runtime_seconds", method_runtime_seconds.get(method, 0.0))
                ),
            }
            per_seed_method_summary.append(average_record)
            method_average_grouped[method].append(average_record)
            per_seed_method_runtime.append(
                {
                    "seed": seed,
                    "method": method,
                    "runtime_seconds": float(average_record["runtime_seconds"]),
                }
            )

        for record in comparison.get("results", []):
            method = str(record["method"])
            variable = str(record["variable"])
            result_record = {
                "seed": seed,
                "method": method,
                "variable": variable,
                "exact_acc": float(record["exact_acc"]),
                "mean_shared_digits": float(record["mean_shared_digits"]),
            }
            per_seed_variable_results.append(result_record)
            variable_grouped[(method, variable)].append(result_record)

    seeds = sorted(set(seeds))
    methods = sorted(methods_seen)

    method_summary = []
    for method in methods:
        method_records = method_average_grouped[method]
        exact_mean, exact_std = _mean_std([record["exact_acc"] for record in method_records])
        shared_mean, shared_std = _mean_std(
            [record["mean_shared_digits"] for record in method_records]
        )
        runtime_mean, runtime_std = _mean_std([record["runtime_seconds"] for record in method_records])
        method_summary.append(
            {
                "method": method,
                "num_seeds": len(method_records),
                "exact_acc_mean": exact_mean,
                "exact_acc_std": exact_std,
                "mean_shared_digits_mean": shared_mean,
                "mean_shared_digits_std": shared_std,
                "runtime_seconds_mean": runtime_mean,
                "runtime_seconds_std": runtime_std,
            }
        )

    variable_summary = []
    for method in methods:
        for variable in target_vars:
            records = variable_grouped.get((method, variable), [])
            exact_mean, exact_std = _mean_std([record["exact_acc"] for record in records])
            shared_mean, shared_std = _mean_std(
                [record["mean_shared_digits"] for record in records]
            )
            variable_summary.append(
                {
                    "method": method,
                    "variable": variable,
                    "num_seeds": len(records),
                    "exact_acc_mean": exact_mean,
                    "exact_acc_std": exact_std,
                    "mean_shared_digits_mean": shared_mean,
                    "mean_shared_digits_std": shared_std,
                }
            )

    backbone_exact_mean, backbone_exact_std = _mean_std(
        [record["exact_acc"] for record in backbone_factual]
    )

    return {
        "seeds": seeds,
        "methods": methods,
        "target_vars": target_vars,
        "seed_runs": seed_runs,
        "backbone_factual_validation": backbone_factual,
        "backbone_factual_validation_summary": {
            "num_seeds": len(backbone_factual),
            "exact_acc_mean": backbone_exact_mean,
            "exact_acc_std": backbone_exact_std,
        },
        "per_seed_method_summary": per_seed_method_summary,
        "per_seed_method_runtime": per_seed_method_runtime,
        "per_seed_variable_results": per_seed_variable_results,
        "method_summary_across_seeds": method_summary,
        "variable_summary_across_seeds": variable_summary,
    }

def _plot_mean_std_bars(
    records: list[dict[str, object]],
    output_path: Path,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
) -> str:
    """Plot one bar per method with cross-seed standard-deviation error bars."""
    methods = [str(record["method"]).upper() for record in records]
    means = [float(record[mean_key]) for record in records]
    stds = [float(record[std_key]) for record in records]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    colors = [
        METHOD_PASTEL_COLORS.get(str(record["method"]).lower(), DEFAULT_PASTEL_COLORS[index % len(DEFAULT_PASTEL_COLORS)])
        for index, record in enumerate(records)
    ]
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="#6b7280", linewidth=0.8)
    ax.set_xticks(x, methods)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def _plot_grouped_mean_std_bars(
    records: list[dict[str, object]],
    output_path: Path,
    group_key: str,
    series_key: str,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    group_order: list[str] | None = None,
    series_order: list[str] | None = None,
) -> str:
    """Plot grouped mean/std bars, such as variables grouped with one bar per method."""
    if group_order is None:
        group_values = sorted({str(record[group_key]) for record in records})
    else:
        group_values = [str(value) for value in group_order]
    if series_order is None:
        series_values = sorted({str(record[series_key]) for record in records})
    else:
        series_values = [str(value) for value in series_order]

    record_map = {
        (str(record[group_key]), str(record[series_key])): record
        for record in records
    }

    x = np.arange(len(group_values), dtype=float)
    width = 0.8 / max(len(series_values), 1)
    offsets = (np.arange(len(series_values), dtype=float) - (len(series_values) - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for index, series in enumerate(series_values):
        means = []
        stds = []
        bar_colors = []
        base_color = METHOD_PASTEL_COLORS.get(
            str(series).lower(),
            DEFAULT_PASTEL_COLORS[index % len(DEFAULT_PASTEL_COLORS)],
        )
        for group in group_values:
            record = record_map.get((group, series))
            means.append(float(record[mean_key]) if record is not None else 0.0)
            stds.append(float(record[std_key]) if record is not None else 0.0)
            bar_colors.append(_shade_color(base_color, len(bar_colors), len(group_values)))
        ax.bar(
            x + offsets[index],
            means,
            width=width,
            yerr=stds,
            capsize=5,
            color=bar_colors,
            edgecolor="#6b7280",
            linewidth=0.8,
            label=str(series).upper(),
        )
    ax.set_xticks(x, group_values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return str(output_path)


def save_seed_sweep_plots(payload: dict[str, object], output_path: str | Path) -> dict[str, str]:
    """Write aggregate multi-seed plots next to the provided output path."""
    output_path = Path(output_path)
    plot_dir = output_path.parent
    ensure_parent_dir(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "plot_dir": str(plot_dir),
        "average_exact_summary": _plot_mean_std_bars(
            records=list(payload.get("method_summary_across_seeds", [])),
            output_path=plot_dir / "average_exact_summary.png",
            mean_key="exact_acc_mean",
            std_key="exact_acc_std",
            ylabel="Average Exact Accuracy",
            title="Average exact accuracy mean +/- std across seeds",
        ),
        "average_shared_summary": _plot_mean_std_bars(
            records=list(payload.get("method_summary_across_seeds", [])),
            output_path=plot_dir / "average_shared_summary.png",
            mean_key="mean_shared_digits_mean",
            std_key="mean_shared_digits_std",
            ylabel="Average Mean Shared Digits",
            title="Average shared-digit mean +/- std across seeds",
        ),
        "variable_exact_summary": _plot_grouped_mean_std_bars(
            records=list(payload.get("variable_summary_across_seeds", [])),
            output_path=plot_dir / "variable_exact_summary.png",
            group_key="variable",
            series_key="method",
            mean_key="exact_acc_mean",
            std_key="exact_acc_std",
            ylabel="Per-Variable Exact Accuracy",
            title="Per-variable exact accuracy mean +/- std across seeds",
            group_order=[str(value) for value in payload.get("target_vars", [])],
            series_order=[str(value) for value in payload.get("methods", [])],
        ),
        "variable_shared_summary": _plot_grouped_mean_std_bars(
            records=list(payload.get("variable_summary_across_seeds", [])),
            output_path=plot_dir / "variable_shared_summary.png",
            group_key="variable",
            series_key="method",
            mean_key="mean_shared_digits_mean",
            std_key="mean_shared_digits_std",
            ylabel="Per-Variable Mean Shared Digits",
            title="Per-variable shared-digit mean +/- std across seeds",
            group_order=[str(value) for value in payload.get("target_vars", [])],
            series_order=[str(value) for value in payload.get("methods", [])],
        ),
        "runtime_summary": _plot_mean_std_bars(
            records=list(payload.get("method_summary_across_seeds", [])),
            output_path=plot_dir / "runtime_summary.png",
            mean_key="runtime_seconds_mean",
            std_key="runtime_seconds_std",
            ylabel="Runtime (s)",
            title="Method runtime mean +/- std across seeds",
        ),
    }
    return plot_paths


def format_seed_sweep_summary(payload: dict[str, object]) -> str:
    """Format a compact text summary of the aggregated multi-seed results."""
    lines = [
        "Addition Seed Sweep Summary",
        f"seeds: {', '.join(str(seed) for seed in payload.get('seeds', []))}",
        "",
        "Backbone Factual Validation",
    ]
    backbone_summary = dict(payload.get("backbone_factual_validation_summary", {}))
    lines.append(
        "mean_exact="
        f"{float(backbone_summary.get('exact_acc_mean', 0.0)):.4f}, "
        "std_exact="
        f"{float(backbone_summary.get('exact_acc_std', 0.0)):.4f}"
    )
    lines.append("")
    lines.append("Method Average Across Seeds")
    for record in payload.get("method_summary_across_seeds", []):
        lines.append(
            f"{str(record['method']).upper()}: "
            f"exact={float(record['exact_acc_mean']):.4f} +/- {float(record['exact_acc_std']):.4f}, "
            f"shared={float(record['mean_shared_digits_mean']):.4f} +/- "
            f"{float(record['mean_shared_digits_std']):.4f}, "
            f"runtime_s={float(record['runtime_seconds_mean']):.2f} +/- "
            f"{float(record['runtime_seconds_std']):.2f}"
        )
    variable_summary = list(payload.get("variable_summary_across_seeds", []))
    if variable_summary:
        lines.append("")
        lines.append("Per-Variable Summary Across Seeds")
        for record in variable_summary:
            lines.append(
                f"{str(record['method']).upper()} [{str(record['variable'])}]: "
                f"exact={float(record['exact_acc_mean']):.4f} +/- "
                f"{float(record['exact_acc_std']):.4f}, "
                f"shared={float(record['mean_shared_digits_mean']):.4f} +/- "
                f"{float(record['mean_shared_digits_std']):.4f}"
            )
    return "\n".join(lines)
