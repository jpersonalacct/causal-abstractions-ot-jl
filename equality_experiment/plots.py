"""Plotting helpers for equality comparison outputs."""

from __future__ import annotations

from pathlib import Path

from . import _env  # noqa: F401

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import DEFAULT_TARGET_VARS
from .runtime import ensure_parent_dir

METHOD_PASTEL_COLORS = {
    "das": "#59a14f",
    "fgw": "#f28e2b",
    "gw": "#4e79a7",
    "ot": "#b07aa1",
    "uot": "#e15759",
}

DEFAULT_PASTEL_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#b07aa1",
    "#e15759",
    "#76b7b2",
)


def get_method_color(method: str, fallback_index: int) -> str:
    """Return a stable per-method color, using a large categorical palette for unknown methods."""
    normalized = str(method).lower()
    explicit = METHOD_PASTEL_COLORS.get(normalized)
    if explicit is not None:
        return explicit
    cmap = plt.get_cmap("tab20")
    return matplotlib.colors.to_hex(cmap(int(fallback_index) % cmap.N))


def _group_records(records: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    grouped = {}
    for record in records:
        method = str(record["method"])
        variable = str(record["variable"])
        grouped.setdefault(method, {})
        grouped[method][variable] = float(record[key])
    return grouped


def save_comparison_plots(
    payload: dict[str, object],
    output_path: str | Path,
    method_payloads: dict[str, dict[str, object]] | None = None,
) -> dict[str, str]:
    """Render the standard equality comparison plots."""
    output_path = Path(output_path)
    plot_dir = output_path.parent
    ensure_parent_dir(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = list(payload.get("results", []))
    summary = list(payload.get("method_summary", []))
    stem = output_path.stem
    prefix = "" if stem == "equality_run_results" else f"{stem}__"

    exact_by_method = _group_records(records, "exact_acc")
    methods = sorted(exact_by_method.keys())
    variables = [str(variable) for variable in payload.get("target_vars", DEFAULT_TARGET_VARS)]
    x = np.arange(len(variables))
    width = 0.8 / max(len(methods), 1)

    exact_path = plot_dir / f"{prefix}exact_accuracy.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for idx, method in enumerate(methods):
        y = [exact_by_method.get(method, {}).get(variable, np.nan) for variable in variables]
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
    ax.set_title("Hierarchical equality per-variable exact accuracy")
    ax.legend(loc="best")
    fig.savefig(exact_path, dpi=200)
    plt.close(fig)

    summary_path = plot_dir / f"{prefix}average_summary.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    summary_methods = [str(record["method"]).upper() for record in summary]
    summary_exact = [float(record["exact_acc"]) for record in summary]
    summary_x = np.arange(len(summary_methods))
    ax.bar(
        summary_x,
        summary_exact,
        width=0.5,
        color="#9ecae1",
        edgecolor="#6b7280",
        linewidth=0.8,
    )
    ax.set_xticks(summary_x, summary_methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Counterfactual Accuracy")
    ax.set_title("Average summary across abstract variables")
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    return {
        "plot_dir": str(plot_dir),
        "exact_accuracy": str(exact_path),
        "average_summary": str(summary_path),
    }
