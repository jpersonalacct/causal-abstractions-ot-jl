"""Console reporting helpers for per-variable and average experiment summaries."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .runtime import ensure_parent_dir


def _format_site_config(record: dict[str, object]) -> str:
    """Format the table's site/config cell for one result record."""
    site_label = str(record.get("site_label", "n/a"))
    if str(record.get("method", "")).lower() != "das":
        return site_label
    epochs_ran = record.get("train_epochs_ran")
    if epochs_ran is None:
        return site_label
    return f"{site_label},e{int(epochs_ran)}"


def print_results_table(records: list[dict[str, object]], title: str) -> None:
    """Print a compact table of per-variable experiment results."""
    print(title)
    if not records:
        print("(no records)")
        return

    header = (
        f"{'method':<8} {'variable':<8} {'exact':>8} {'shared':>8} {'select/cal':>10} {'site/config':<24}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        print(
            f"{str(record['method']):<8} "
            f"{str(record.get('variable', 'average')):<8} "
            f"{float(record['exact_acc']):>8.4f} "
            f"{float(record['mean_shared_digits']):>8.4f} "
            f"{float(record.get('selection_exact_acc', 0.0)):>10.4f} "
            f"{_format_site_config(record):<24}"
        )


def summarize_method_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Average per-variable metrics into one average summary per method."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)

    summaries = []
    for method, method_records in sorted(grouped.items()):
        exact_avg = sum(float(record["exact_acc"]) for record in method_records) / len(method_records)
        shared_avg = (
            sum(float(record["mean_shared_digits"]) for record in method_records) / len(method_records)
        )
        summaries.append(
            {
                "method": method,
                "exact_acc": exact_avg,
                "mean_shared_digits": shared_avg,
            }
        )
    return summaries


def build_method_selection_summary(method: str, payload: dict[str, object]) -> dict[str, object]:
    """Build a compact per-method summary for saved artifacts."""
    if method in {"gw", "ot", "fgw"}:
        return {
            "method": method,
            "transport_meta": dict(payload.get("transport_meta", {})),
            "selected_hyperparameters": dict(payload.get("selected_hyperparameters", {})),
            "selection_objective": payload.get("selection_objective"),
            "final_evaluation_policy": payload.get("final_evaluation_policy"),
            "results": [
                {
                    "variable": record["variable"],
                    "site_label": record["site_label"],
                    "top_site_label": record.get("top_site_label"),
                    "selected_layer": record.get("selected_layer"),
                    "continuous_cutoff": record.get("continuous_cutoff"),
                    "selection_cross_entropy": record.get("selection_cross_entropy"),
                    "top_k": record.get("top_k"),
                    "lambda": record.get("lambda"),
                    "calibration_exact_acc": record.get("calibration_exact_acc", 0.0),
                    "calibration_mean_shared_digits": record.get("calibration_mean_shared_digits", 0.0),
                    "exact_acc": record["exact_acc"],
                    "mean_shared_digits": record["mean_shared_digits"],
                }
                for record in payload.get("results", [])
            ],
        }

    if method == "das":
        return {
            "method": method,
            "training_stopping_rule": dict(payload.get("training_stopping_rule", {})),
            "results": [
                {
                    "variable": record["variable"],
                    "site_label": record["site_label"],
                    "layer": record.get("layer"),
                    "subspace_dim": record.get("subspace_dim"),
                    "train_epochs_ran": record.get("train_epochs_ran"),
                    "calibration_exact_acc": record.get("calibration_exact_acc", 0.0),
                    "calibration_mean_shared_digits": record.get("calibration_mean_shared_digits", 0.0),
                    "exact_acc": record["exact_acc"],
                    "mean_shared_digits": record["mean_shared_digits"],
                }
                for record in payload.get("results", [])
            ],
        }

    raise ValueError(f"Unsupported method summary type: {method}")


def format_method_selection_summary(summary: dict[str, object]) -> str:
    """Format one compact method summary as human-readable text."""
    method = str(summary["method"]).upper()
    lines = [method]

    if method in {"GW", "OT", "FGW"}:
        transport_meta = dict(summary.get("transport_meta", {}))
        selected_hyperparameters = dict(summary.get("selected_hyperparameters", {}))
        selection_objective = summary.get("selection_objective")
        final_evaluation_policy = summary.get("final_evaluation_policy")
        if transport_meta:
            solver_bits = ", ".join(f"{key}={value}" for key, value in sorted(transport_meta.items()))
            lines.append(f"solver: {solver_bits}")
        if selection_objective is not None:
            lines.append(f"selection_objective: {selection_objective}")
        if final_evaluation_policy is not None:
            lines.append(f"final_evaluation_policy: {final_evaluation_policy}")
        lines.append("selected soft matches:")
        for record in summary.get("results", []):
            variable = str(record["variable"])
            top_k = record.get("top_k")
            lambda_value = record.get("lambda")
            selected_layer = record.get("selected_layer")
            continuous_cutoff = record.get("continuous_cutoff")
            if isinstance(selected_hyperparameters.get("top_k_by_variable"), dict):
                top_k = selected_hyperparameters["top_k_by_variable"].get(variable, top_k)
            if isinstance(selected_hyperparameters.get("lambda_by_variable"), dict):
                lambda_value = selected_hyperparameters["lambda_by_variable"].get(variable, lambda_value)
            if isinstance(selected_hyperparameters.get("selected_layer_by_variable"), dict):
                selected_layer = selected_hyperparameters["selected_layer_by_variable"].get(variable, selected_layer)
            if isinstance(selected_hyperparameters.get("continuous_cutoff_by_variable"), dict):
                continuous_cutoff = selected_hyperparameters["continuous_cutoff_by_variable"].get(
                    variable,
                    continuous_cutoff,
                )
            lines.append(
                f"{variable}: layer={selected_layer}, top_k={top_k}, cutoff={continuous_cutoff}, "
                f"lambda={lambda_value}, top_site={record.get('top_site_label')}, "
                f"cal_exact={float(record.get('calibration_exact_acc', 0.0)):.4f}, "
                f"cal_shared={float(record.get('calibration_mean_shared_digits', 0.0)):.4f}, "
                f"test_exact={float(record['exact_acc']):.4f}, "
                f"test_shared={float(record['mean_shared_digits']):.4f}"
            )
        return "\n".join(lines)

    training_stopping_rule = dict(summary.get("training_stopping_rule", {}))
    if training_stopping_rule:
        display_rule = {
            key: value for key, value in training_stopping_rule.items() if key != "max_epochs"
        }
        stopping_bits = ", ".join(f"{key}={value}" for key, value in sorted(display_rule.items()))
        lines.append(f"training: {stopping_bits}")
    lines.append("selected sites:")
    for record in summary.get("results", []):
        lines.append(
            f"{record['variable']}: site={record['site_label']}, layer={record.get('layer')}, "
            f"subspace_dim={record.get('subspace_dim')}, epochs_ran={record.get('train_epochs_ran')}, "
            f"cal_exact={float(record.get('calibration_exact_acc', 0.0)):.4f}, "
            f"cal_shared={float(record.get('calibration_mean_shared_digits', 0.0)):.4f}, "
            f"test_exact={float(record['exact_acc']):.4f}, "
            f"test_shared={float(record['mean_shared_digits']):.4f}"
        )
    return "\n".join(lines)


def format_method_candidate_sweep(method: str, payload: dict[str, object]) -> str:
    """Format only the candidate-sweep section for one method."""
    lines = [str(method).upper()]
    if method in {"gw", "ot", "fgw"}:
        layer_candidate_summaries = dict(payload.get("layer_candidate_summaries", {}))
        if layer_candidate_summaries:
            lines.append("layer candidate sweep:")
            summary = build_method_selection_summary(method, payload)
            selected_hyperparameters = dict(summary.get("selected_hyperparameters", {}))
            selected_layer_by_variable = dict(selected_hyperparameters.get("selected_layer_by_variable", {}))
            selected_top_k_by_variable = dict(selected_hyperparameters.get("top_k_by_variable", {}))
            selected_lambda_by_variable = dict(selected_hyperparameters.get("lambda_by_variable", {}))
            for variable in payload.get("target_vars", []):
                variable_key = str(variable)
                lines.append(f"{variable_key}:")
                candidates = list(layer_candidate_summaries.get(variable_key, []))
                selected_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.get("layer", -1)) == int(selected_layer_by_variable.get(variable_key, -1))
                ]
                remaining_candidates = [candidate for candidate in candidates if candidate not in selected_candidates]
                for candidate in [*selected_candidates, *remaining_candidates]:
                    selected_marker = " [selected]" if candidate in selected_candidates else ""
                    lines.append(
                        f"{variable_key}{selected_marker}: layer={candidate.get('layer')}, "
                        f"cutoff={float(candidate.get('continuous_cutoff', 0.0)):.4f}, "
                        f"top_k={selected_top_k_by_variable.get(variable_key)}, "
                        f"lambda={selected_lambda_by_variable.get(variable_key)}, "
                        f"top_site={candidate.get('top_site_label', 'n/a')}, "
                        f"cal_ce={float(candidate.get('selection_cross_entropy', 0.0)):.4f}, "
                        f"cal_exact={float(candidate.get('selection_exact_acc', 0.0)):.4f}, "
                        f"cal_shared={float(candidate.get('selection_mean_shared_digits', 0.0)):.4f}"
                    )
            return "\n".join(lines)
        calibration_sweep = dict(payload.get("calibration_sweep", {}))
        if calibration_sweep:
            lines.append("calibration sweep:")
            summary = build_method_selection_summary(method, payload)
            selected_hyperparameters = dict(summary.get("selected_hyperparameters", {}))
            selected_top_k_by_variable = dict(selected_hyperparameters.get("top_k_by_variable", {}))
            selected_lambda_by_variable = dict(selected_hyperparameters.get("lambda_by_variable", {}))
            for variable in payload.get("target_vars", []):
                variable_key = str(variable)
                lines.append(f"{variable_key}:")
                candidates = list(calibration_sweep.get(variable_key, []))
                selected_candidates = [
                    candidate
                    for candidate in candidates
                    if int(candidate.get("top_k", -1)) == int(selected_top_k_by_variable.get(variable_key, -1))
                    and float(candidate.get("lambda", float("nan")))
                    == float(selected_lambda_by_variable.get(variable_key, float("nan")))
                ]
                remaining_candidates = [
                    candidate for candidate in candidates if candidate not in selected_candidates
                ]
                for candidate in [*selected_candidates, *remaining_candidates]:
                    result = dict(candidate.get("result", {}))
                    top_site_label = result.get("top_site_label", "n/a")
                    selected_marker = " [selected]" if candidate in selected_candidates else ""
                    lines.append(
                        f"{variable_key}{selected_marker}: top_k={int(candidate['top_k'])}, "
                        f"lambda={float(candidate['lambda']):g}, top_site={top_site_label}, "
                        f"cal_exact={float(candidate['exact_acc']):.4f}, "
                        f"cal_shared={float(candidate['mean_shared_digits']):.4f}"
                    )
        return "\n".join(lines) if len(lines) > 1 else ""

    search_records = dict(payload.get("search_records", {}))
    if search_records:
        lines.append("candidate sweep:")
        selected_records = {str(record["variable"]): record for record in payload.get("results", [])}
        for variable in payload.get("target_vars", []):
            variable_key = str(variable)
            lines.append(f"{variable_key}:")
            selected_record = selected_records.get(variable_key, {})
            candidates = list(search_records.get(variable_key, []))

            def is_selected(candidate: dict[str, object]) -> bool:
                return (
                    str(candidate.get("site_label")) == str(selected_record.get("site_label"))
                    and int(candidate.get("layer", -1)) == int(selected_record.get("layer", -1))
                    and int(candidate.get("subspace_dim", -1)) == int(selected_record.get("subspace_dim", -1))
                )

            selected_candidates = [candidate for candidate in candidates if is_selected(candidate)]
            remaining_candidates = [candidate for candidate in candidates if not is_selected(candidate)]
            for candidate in [*selected_candidates, *remaining_candidates]:
                loss_history = list(candidate.get("train_loss_history", []))
                final_train_loss = float(loss_history[-1]) if loss_history else 0.0
                selected_marker = " [selected]" if candidate in selected_candidates else ""
                lines.append(
                    f"{variable_key}{selected_marker}: site={candidate['site_label']}, "
                    f"layer={candidate.get('layer')}, subspace_dim={candidate.get('subspace_dim')}, "
                    f"epochs_ran={candidate.get('train_epochs_ran')}, "
                    f"train_loss={final_train_loss:.4f}, "
                    f"cal_exact={float(candidate.get('calibration_exact_acc', 0.0)):.4f}, "
                    f"cal_shared={float(candidate.get('calibration_mean_shared_digits', 0.0)):.4f}"
                )
    return "\n".join(lines) if len(lines) > 1 else ""


def write_text_report(path: str | Path, text: str) -> None:
    """Write a plain-text report to disk."""
    ensure_parent_dir(path)
    Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")
