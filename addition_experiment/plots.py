"""Plotting helpers for comparison outputs written by the experiment scripts."""

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
    "das": "#a8d5ba",
    "fgw": "#f6bd60",
    "gw": "#9ecae1",
    "ot": "#cdb4db",
}

SUMMARY_PASTEL_COLORS = {
    "exact": "#9ecae1",
    "shared": "#f7c59f",
}

DEFAULT_PASTEL_COLORS = (
    "#9ecae1",
    "#f6bd60",
    "#a8d5ba",
    "#cdb4db",
    "#f7cad0",
    "#bde0fe",
)


def _group_records(records: list[dict[str, object]], key: str) -> dict[str, dict[str, float]]:
    """Group result values by method and abstract variable."""
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
    """Render the standard comparison plots and return their output paths."""
    output_path = Path(output_path)
    plot_dir = output_path.parent
    ensure_parent_dir(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = list(payload.get("results", []))
    summary = list(payload.get("method_summary", []))

    exact_by_method = _group_records(records, "exact_acc")
    shared_by_method = _group_records(records, "mean_shared_digits")
    methods = sorted(exact_by_method.keys())
    variables = [str(variable) for variable in payload.get("target_vars", DEFAULT_TARGET_VARS)]

    x = np.arange(len(variables))
    width = 0.8 / max(len(methods), 1)

    exact_path = plot_dir / "exact_accuracy.png"
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for idx, method in enumerate(methods):
        y = [exact_by_method.get(method, {}).get(variable, np.nan) for variable in variables]
        ax.bar(
            x + (idx - (len(methods) - 1) / 2.0) * width,
            y,
            width=width,
            color=METHOD_PASTEL_COLORS.get(method.lower(), DEFAULT_PASTEL_COLORS[idx % len(DEFAULT_PASTEL_COLORS)]),
            edgecolor="#6b7280",
            linewidth=0.8,
            label=method.upper(),
        )
    ax.set_xticks(x, variables)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Exact Counterfactual Accuracy")
    ax.set_title("Per-variable exact accuracy")
    ax.legend(loc="best")
    fig.savefig(exact_path, dpi=200)
    plt.close(fig)

    shared_path = plot_dir / "shared_digits.png"
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for idx, method in enumerate(methods):
        y = [shared_by_method.get(method, {}).get(variable, np.nan) for variable in variables]
        ax.bar(
            x + (idx - (len(methods) - 1) / 2.0) * width,
            y,
            width=width,
            color=METHOD_PASTEL_COLORS.get(method.lower(), DEFAULT_PASTEL_COLORS[idx % len(DEFAULT_PASTEL_COLORS)]),
            edgecolor="#6b7280",
            linewidth=0.8,
            label=method.upper(),
        )
    ax.set_xticks(x, variables)
    ax.set_ylim(0.0, 3.0)
    ax.set_ylabel("Mean Shared Digits")
    ax.set_title("Per-variable digit-overlap score")
    ax.legend(loc="best")
    fig.savefig(shared_path, dpi=200)
    plt.close(fig)

    summary_path = plot_dir / "average_summary.png"
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    summary_methods = [str(record["method"]).upper() for record in summary]
    summary_exact = [float(record["exact_acc"]) for record in summary]
    summary_shared = [float(record["mean_shared_digits"]) for record in summary]
    summary_x = np.arange(len(summary_methods))
    ax.bar(
        summary_x - 0.18,
        summary_exact,
        width=0.36,
        color=SUMMARY_PASTEL_COLORS["exact"],
        edgecolor="#6b7280",
        linewidth=0.8,
        label="Exact",
    )
    ax.bar(
        summary_x + 0.18,
        summary_shared,
        width=0.36,
        color=SUMMARY_PASTEL_COLORS["shared"],
        edgecolor="#6b7280",
        linewidth=0.8,
        label="Shared digits",
    )
    ax.set_xticks(summary_x, summary_methods)
    ax.set_title("Average summary across abstract variables")
    ax.legend(loc="best")
    fig.savefig(summary_path, dpi=200)
    plt.close(fig)

    plot_paths = {
        "plot_dir": str(plot_dir),
        "exact_accuracy": str(exact_path),
        "shared_digits": str(shared_path),
        "average_summary": str(summary_path),
    }

    transport_methods = []
    if method_payloads is not None:
        for method in ("gw", "ot", "fgw"):
            method_payload = method_payloads.get(method)
            if method_payload is not None and "transport" in method_payload:
                transport_methods.append((method, method_payload))
    if transport_methods:
        vmax = max(
            float(np.asarray(method_payload["transport"], dtype=float).max())
            for _, method_payload in transport_methods
        )
        transport_path = plot_dir / "transport_plans.png"
        fig, axes = plt.subplots(
            nrows=len(transport_methods),
            ncols=1,
            figsize=(10, 2.8 * len(transport_methods)),
            constrained_layout=True,
            squeeze=False,
        )
        variables = [str(variable) for variable in payload.get("target_vars", DEFAULT_TARGET_VARS)]
        for axis, (method, method_payload) in zip(axes.flat, transport_methods):
            transport = np.asarray(method_payload["transport"], dtype=float)
            image = axis.imshow(
                transport,
                aspect="auto",
                interpolation="nearest",
                cmap="GnBu",
                vmin=0.0,
                vmax=vmax if vmax > 0.0 else None,
            )
            axis.set_title(f"{method.upper()} transport plan")
            axis.set_ylabel("Abstract variable")
            axis.set_yticks(np.arange(len(variables)), variables)
            axis.set_xlabel("Neural site")
            fig.colorbar(image, ax=axis, shrink=0.9)
        fig.savefig(transport_path, dpi=200)
        plt.close(fig)
        plot_paths["transport_plans"] = str(transport_path)

    return plot_paths
