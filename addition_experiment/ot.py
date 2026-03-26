"""Transport-based alignment methods: OT, GW, and fused GW."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
import torch
import torch.nn.functional as F

from . import _env  # noqa: F401

from pyvene import VanillaIntervention
from scipy.spatial.distance import cdist
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback only when tqdm is unavailable
    tqdm = None

from variable_width_mlp import VariableWidthMLPForClassification, logits_from_output

from .constants import (
    DEFAULT_ALIGNMENT_RESOLUTION,
    DEFAULT_TARGET_VARS,
)
from .metrics import metrics_from_logits
from .pair_bank import PairBank
from .pyvene_utils import CanonicalSite, build_intervenable, enumerate_canonical_sites, run_intervenable_logits


def _format_hparam_value(value: float, decimals: int = 6) -> str:
    """Format sweep hyperparameters with compact but stable decimal output."""
    text = f"{float(value):.{int(decimals)}f}".rstrip("0")
    if text.endswith("."):
        return text + "0"
    return text


@dataclass(frozen=True)
class OTConfig:
    """Hyperparameters for OT, GW, and FGW alignment and intervention runs."""

    method: str = "gw"
    batch_size: int = 128
    geometry_metric: str = "cosine"
    normalize_cost_matrices: bool = True
    epsilon: float = 5e-2
    max_iter: int = 500
    tol: float = 1e-9
    verbose: bool = False
    epsilon_retry_multipliers: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0, 100.0)
    ranking_k: int = 5
    resolution: int = DEFAULT_ALIGNMENT_RESOLUTION
    alpha: float = 0.5
    tau: float = 1.0
    target_vars: tuple[str, ...] = DEFAULT_TARGET_VARS
    top_k_values: tuple[int, ...] | None = None
    lambda_values: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
    selection_verbose: bool = True
    calibration_progress_interval: int = 250


def collect_base_logits(
    model: VariableWidthMLPForClassification,
    base_inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect factual logits for a bank of base inputs."""
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, base_inputs.shape[0], batch_size):
            end = min(start + batch_size, base_inputs.shape[0])
            output = model(inputs_embeds=base_inputs[start:end].to(device).unsqueeze(1))
            chunks.append(logits_from_output(output).detach().cpu())
    return torch.cat(chunks, dim=0)


def collect_site_signatures(
    model: VariableWidthMLPForClassification,
    bank: PairBank,
    sites: list[CanonicalSite],
    base_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Measure each canonical site's intervention effect signature."""
    signatures = []
    base_prob = torch.softmax(base_logits, dim=-1)
    for site in sites:
        intervenable = build_intervenable(
            model=model,
            layer=site.layer,
            component=site.component,
            intervention=VanillaIntervention(),
            device=device,
            unit=site.unit,
            max_units=site.max_units,
            freeze_model=True,
            freeze_intervention=True,
            use_fast=False,
        )
        site_logits = run_intervenable_logits(
            intervenable=intervenable,
            base_inputs=bank.base_inputs,
            source_inputs=bank.source_inputs,
            subspace_dims=site.subspace_dims,
            position=site.position,
            batch_size=batch_size,
            device=device,
        )
        site_effect = (torch.softmax(site_logits, dim=-1) - base_prob).permute(1, 0).contiguous()
        signatures.append(site_effect.reshape(-1))
    return torch.stack(signatures, dim=0)


def build_variable_signatures(
    bank: PairBank,
    num_classes: int,
    target_vars: tuple[str, ...],
) -> torch.Tensor:
    """Build abstract-variable effect signatures from SCM labels."""
    base_onehot = F.one_hot(bank.base_labels, num_classes=num_classes).to(torch.float32)
    signatures = []
    for variable in target_vars:
        cf_onehot = F.one_hot(bank.cf_labels_by_var[variable], num_classes=num_classes).to(torch.float32)
        effect = (cf_onehot - base_onehot).permute(1, 0).contiguous()
        signatures.append(effect.reshape(-1))
    return torch.stack(signatures, dim=0)


def build_geometry_costs(
    variable_signatures: torch.Tensor,
    site_signatures: torch.Tensor,
    metric: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute abstract-space and neural-site relational cost matrices."""
    variable_np = variable_signatures.cpu().numpy()
    site_np = site_signatures.cpu().numpy()
    cost_var = cdist(variable_np, variable_np, metric=metric)
    cost_site = cdist(site_np, site_np, metric=metric)

    cost_var = np.nan_to_num(cost_var, nan=0.0, posinf=0.0, neginf=0.0)
    cost_site = np.nan_to_num(cost_site, nan=0.0, posinf=0.0, neginf=0.0)
    cost_var = 0.5 * (cost_var + cost_var.T)
    cost_site = 0.5 * (cost_site + cost_site.T)
    np.fill_diagonal(cost_var, 0.0)
    np.fill_diagonal(cost_site, 0.0)

    if normalize:
        if float(cost_var.max()) > 0:
            cost_var = cost_var / float(cost_var.max())
        if float(cost_site.max()) > 0:
            cost_site = cost_site / float(cost_site.max())
    return cost_var, cost_site


def build_cross_cost(
    variable_signatures: torch.Tensor,
    site_signatures: torch.Tensor,
    metric: str,
    normalize: bool,
) -> np.ndarray:
    """Compute the direct abstract-to-site alignment cost matrix."""
    cost_cross = cdist(variable_signatures.cpu().numpy(), site_signatures.cpu().numpy(), metric=metric)
    cost_cross = np.nan_to_num(cost_cross, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize and float(cost_cross.max()) > 0:
        cost_cross = cost_cross / float(cost_cross.max())
    return cost_cross


def solve_gw_transport(
    cost_var: np.ndarray,
    cost_site: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the entropic Gromov-Wasserstein transport problem."""
    last_transport = None
    last_log = None
    for multiplier in config.epsilon_retry_multipliers:
        epsilon = config.epsilon * config.tau * multiplier
        transport, log = ot.gromov.entropic_gromov_wasserstein(
            cost_var,
            cost_site,
            p,
            q,
            loss_fun="square_loss",
            epsilon=epsilon,
            max_iter=config.max_iter,
            tol=config.tol,
            log=True,
            verbose=config.verbose,
        )
        last_transport = transport
        last_log = log
        if np.isfinite(transport).all() and float(np.sum(transport)) > 0.0:
            return transport, {"method": "gw", "epsilon_used": epsilon, "tau_used": config.tau}
    return last_transport, {"method": "gw_degenerate", "tau_used": config.tau, "log": last_log}


def solve_ot_transport(
    cost_cross: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the entropic optimal transport problem on direct costs."""
    last_transport = None
    for multiplier in config.epsilon_retry_multipliers:
        epsilon = config.epsilon * config.tau * multiplier
        transport = ot.sinkhorn(
            p,
            q,
            cost_cross,
            reg=epsilon,
            numItermax=config.max_iter,
            stopThr=config.tol,
            verbose=config.verbose,
        )
        last_transport = transport
        if np.isfinite(transport).all() and float(np.sum(transport)) > 0.0:
            return transport, {"method": "ot", "epsilon_used": epsilon, "tau_used": config.tau}
    return last_transport, {"method": "ot_degenerate", "tau_used": config.tau}


def solve_fgw_transport(
    cost_cross: np.ndarray,
    cost_var: np.ndarray,
    cost_site: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    config: OTConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Solve the fused Gromov-Wasserstein transport problem."""
    last_transport = None
    last_log = None
    for multiplier in config.epsilon_retry_multipliers:
        epsilon = config.epsilon * config.tau * multiplier
        transport, log = ot.gromov.BAPG_fused_gromov_wasserstein(
            cost_cross,
            cost_var,
            cost_site,
            p=p,
            q=q,
            loss_fun="square_loss",
            epsilon=epsilon,
            alpha=config.alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=config.verbose,
            log=True,
        )
        last_transport = transport
        last_log = log
        if np.isfinite(transport).all() and float(np.sum(transport)) > 0.0:
            return transport, {"method": "fgw", "epsilon_used": epsilon, "tau_used": config.tau, "alpha": config.alpha}
    return last_transport, {"method": "fgw_degenerate", "tau_used": config.tau, "alpha": config.alpha, "log": last_log}


def build_rankings(
    transport: np.ndarray,
    sites: list[CanonicalSite],
    target_vars: tuple[str, ...],
    ranking_k: int,
) -> dict[str, list[dict[str, object]]]:
    """Rank candidate neural sites for each abstract variable by transport mass."""
    rankings = {}
    for variable_index, variable in enumerate(target_vars):
        order = np.argsort(-transport[variable_index])[:ranking_k]
        rankings[variable] = [
            {
                "site_index": int(site_index),
                "layer": int(sites[int(site_index)].layer),
                "dims": list(sites[int(site_index)].dims),
                "site_label": sites[int(site_index)].label,
                "transport_mass": float(transport[variable_index, int(site_index)]),
            }
            for site_index in order
        ]
    return rankings


def normalize_transport_rows(transport: np.ndarray) -> np.ndarray:
    """Normalize each transport row into a per-variable distribution over sites."""
    row_sums = transport.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return transport / safe_row_sums


def resolve_top_k_values(top_k_values: tuple[int, ...] | None, num_sites: int) -> tuple[int, ...]:
    """Resolve the top-k sweep values used during calibration selection."""
    if num_sites <= 0:
        raise ValueError(f"num_sites must be positive, got {num_sites}")
    if top_k_values is None:
        return tuple(range(1, num_sites + 1))

    resolved = []
    for raw_value in top_k_values:
        value = min(max(int(raw_value), 1), num_sites)
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError("top_k_values must contain at least one positive value")
    return tuple(resolved)


def truncate_transport_rows(
    normalized_transport: np.ndarray,
    top_k: int | tuple[int, ...] | list[int],
    renormalize: bool = False,
) -> np.ndarray:
    """Keep only top-k sites in each row and optionally renormalize the retained mass."""
    if isinstance(top_k, int):
        top_k_by_row = [max(1, min(int(top_k), normalized_transport.shape[1]))] * normalized_transport.shape[0]
    else:
        top_k_by_row = [max(1, min(int(value), normalized_transport.shape[1])) for value in top_k]
        if len(top_k_by_row) != normalized_transport.shape[0]:
            raise ValueError(
                f"Expected one top-k value per row, got {len(top_k_by_row)} values "
                f"for {normalized_transport.shape[0]} rows"
            )

    truncated = np.zeros_like(normalized_transport)
    for variable_index in range(normalized_transport.shape[0]):
        order = np.argsort(-normalized_transport[variable_index], kind="stable")[: top_k_by_row[variable_index]]
        truncated[variable_index, order] = normalized_transport[variable_index, order]
        if renormalize:
            row_sum = float(truncated[variable_index].sum())
            if row_sum > 0.0:
                truncated[variable_index] = truncated[variable_index] / row_sum
    return truncated


def build_layer_masks_from_transport(
    model: VariableWidthMLPForClassification,
    sites: list[CanonicalSite],
    normalized_transport: np.ndarray,
    target_vars: tuple[str, ...],
) -> dict[str, dict[int, torch.Tensor]]:
    """Convert row-normalized transport weights into per-layer neuron masks."""
    layer_masks_by_variable = {}
    for variable_index, variable in enumerate(target_vars):
        layer_masks = {
            layer: torch.zeros(int(model.config.hidden_dims[layer]), dtype=torch.float32)
            for layer in range(model.config.n_layer)
        }
        for site_index, site in enumerate(sites):
            weight = float(normalized_transport[variable_index, site_index])
            if weight <= 0.0:
                continue
            per_dim_weight = weight / float(len(site.dims))
            for dim in site.dims:
                layer_masks[site.layer][dim] += per_dim_weight
        layer_masks_by_variable[variable] = {
            layer: mask for layer, mask in layer_masks.items() if float(mask.sum().item()) > 0.0
        }
    return layer_masks_by_variable


def run_soft_transport_intervention_logits(
    model: VariableWidthMLPForClassification,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    layer_masks: dict[int, torch.Tensor],
    strength: float,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Apply the soft transport intervention across all hidden layers."""
    outputs = []
    device = torch.device(device)
    device_masks = {
        layer: mask.to(device=device, dtype=torch.float32).view(1, 1, -1) for layer, mask in layer_masks.items()
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, base_inputs.shape[0], batch_size):
            end = min(start + batch_size, base_inputs.shape[0])
            base_hidden = base_inputs[start:end].to(device)
            source_hidden = source_inputs[start:end].to(device)
            if base_hidden.ndim == 2:
                base_hidden = base_hidden.unsqueeze(1)
            if source_hidden.ndim == 2:
                source_hidden = source_hidden.unsqueeze(1)

            for layer, block in enumerate(model.h):
                source_hidden = block(source_hidden)
                base_hidden = block(base_hidden)
                layer_mask = device_masks.get(layer)
                if layer_mask is not None:
                    base_hidden = base_hidden + float(strength) * layer_mask * (source_hidden - base_hidden)

            logits = model.score(base_hidden)
            if model.config.squeeze_output:
                logits = logits.squeeze(1)
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_soft_transport_interventions(
    method_name: str,
    model: VariableWidthMLPForClassification,
    evaluation_bank: PairBank,
    sites: list[CanonicalSite],
    transport_weights: np.ndarray,
    rankings: dict[str, list[dict[str, object]]],
    target_vars: tuple[str, ...],
    top_k_by_variable: dict[str, int],
    lambda_by_variable: dict[str, float],
    batch_size: int,
    device: torch.device | str,
) -> tuple[list[dict[str, object]], dict[str, dict[int, torch.Tensor]]]:
    """Evaluate the soft transport intervention induced by each transport row."""
    layer_masks_by_variable = build_layer_masks_from_transport(
        model,
        sites,
        transport_weights,
        target_vars,
    )
    records = []
    for variable_index, variable in enumerate(target_vars):
        top_k = int(top_k_by_variable[variable])
        strength = float(lambda_by_variable[variable])
        logits = run_soft_transport_intervention_logits(
            model=model,
            base_inputs=evaluation_bank.base_inputs,
            source_inputs=evaluation_bank.source_inputs,
            layer_masks=layer_masks_by_variable[variable],
            strength=strength,
            batch_size=batch_size,
            device=device,
        )
        top_site = rankings[variable][0] if rankings[variable] else None
        records.append(
            {
                "method": method_name,
                "variable": variable,
                "split": evaluation_bank.split,
                "seed": evaluation_bank.seed,
                "site_label": f"soft:k{int(top_k)},l{_format_hparam_value(float(strength))}",
                "top_k": int(top_k),
                "lambda": float(strength),
                "top_site_label": top_site["site_label"] if top_site is not None else None,
                "top_site_transport_mass": (
                    float(transport_weights[variable_index, int(top_site["site_index"])]) if top_site else 0.0
                ),
                "layer_mass_by_layer": {
                    f"L{layer}": float(float(strength) * mask.sum().item())
                    for layer, mask in layer_masks_by_variable[variable].items()
                },
                **metrics_from_logits(logits, evaluation_bank.cf_labels_by_var[variable]),
            }
        )
    return records, layer_masks_by_variable


def summarize_candidate_records(records: list[dict[str, object]]) -> dict[str, float]:
    """Compute average metrics for one calibration candidate."""
    if not records:
        return {"exact_acc": 0.0, "mean_shared_digits": 0.0}
    exact_acc = sum(float(record["exact_acc"]) for record in records) / len(records)
    mean_shared_digits = sum(float(record["mean_shared_digits"]) for record in records) / len(records)
    return {
        "exact_acc": exact_acc,
        "mean_shared_digits": mean_shared_digits,
    }


def choose_better_variable_candidate(
    candidate: dict[str, object],
    incumbent: dict[str, object] | None,
) -> bool:
    """Decide whether one per-variable calibration candidate beats the current incumbent."""
    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["exact_acc"]),
        float(candidate["mean_shared_digits"]),
    )
    incumbent_key = (
        float(incumbent["exact_acc"]),
        float(incumbent["mean_shared_digits"]),
    )
    return candidate_key > incumbent_key


def select_transport_hyperparameters(
    method_name: str,
    model: VariableWidthMLPForClassification,
    calibration_bank: PairBank,
    sites: list[CanonicalSite],
    normalized_transport: np.ndarray,
    rankings: dict[str, list[dict[str, object]]],
    target_vars: tuple[str, ...],
    batch_size: int,
    device: torch.device | str,
    config: OTConfig,
) -> dict[str, object]:
    """Select the best (top-k, lambda) pair independently for each variable."""
    top_k_values = resolve_top_k_values(config.top_k_values, len(sites))
    candidates_per_variable = len(top_k_values) * len(config.lambda_values)
    total_candidates = len(target_vars) * candidates_per_variable

    if config.selection_verbose:
        print(
            f"{method_name.upper()} calibration "
            f"| candidates_per_variable={candidates_per_variable} "
            f"| total_candidates={total_candidates} "
            f"| top_k_values={top_k_values} "
            f"| lambdas={tuple(float(value) for value in config.lambda_values)}"
        )

    selected_top_k_by_variable = {}
    selected_lambda_by_variable = {}
    selected_results = []
    search_records = {}

    for variable_index, variable in enumerate(target_vars):
        variable_transport = normalized_transport[variable_index : variable_index + 1]
        variable_rankings = {variable: rankings[variable]}
        best_candidate = None
        variable_candidates = []
        top_site_label = rankings[variable][0]["site_label"] if rankings[variable] else "n/a"
        progress_bar = None
        if config.selection_verbose and tqdm is not None:
            progress_bar = tqdm(
                total=candidates_per_variable,
                desc=f"{method_name.upper()} [{variable}]",
                unit="candidate",
                dynamic_ncols=True,
                leave=True,
            )
        try:
            for top_k in top_k_values:
                truncated_transport = truncate_transport_rows(
                    variable_transport,
                    top_k,
                    renormalize=True,
                )
                for strength in config.lambda_values:
                    calibration_results, _ = evaluate_soft_transport_interventions(
                        method_name=method_name,
                        model=model,
                        evaluation_bank=calibration_bank,
                        sites=sites,
                        transport_weights=truncated_transport,
                        rankings=variable_rankings,
                        target_vars=(variable,),
                        top_k_by_variable={variable: int(top_k)},
                        lambda_by_variable={variable: float(strength)},
                        batch_size=batch_size,
                        device=device,
                    )
                    calibration_record = calibration_results[0]
                    candidate = {
                        "variable": variable,
                        "top_k": int(top_k),
                        "lambda": float(strength),
                        "exact_acc": float(calibration_record["exact_acc"]),
                        "mean_shared_digits": float(calibration_record["mean_shared_digits"]),
                        "result": calibration_record,
                    }
                    variable_candidates.append(candidate)
                    if choose_better_variable_candidate(candidate, best_candidate):
                        best_candidate = candidate
                    if progress_bar is not None:
                        progress_bar.update(1)
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if best_candidate is None:
            raise RuntimeError(f"Failed to select transport hyperparameters for {method_name}:{variable}")

        selected_top_k_by_variable[variable] = int(best_candidate["top_k"])
        selected_lambda_by_variable[variable] = float(best_candidate["lambda"])
        selected_results.append(best_candidate["result"])
        search_records[variable] = variable_candidates

        if config.selection_verbose:
            print(
                f"{method_name.upper()} [{variable}] calibration best "
                f"| top_k={int(best_candidate['top_k'])} "
                f"| lambda={_format_hparam_value(float(best_candidate['lambda']))} "
                f"| top_site={top_site_label} "
                f"| exact={float(best_candidate['exact_acc']):.4f} "
                f"| shared={float(best_candidate['mean_shared_digits']):.4f}"
            )

    average_metrics = summarize_candidate_records(selected_results)
    return {
        "selected_top_k_by_variable": selected_top_k_by_variable,
        "selected_lambda_by_variable": selected_lambda_by_variable,
        "results": selected_results,
        "search_records": search_records,
        "average_calibration_exact_acc": average_metrics["exact_acc"],
        "average_calibration_mean_shared_digits": average_metrics["mean_shared_digits"],
    }


def run_alignment_pipeline(
    model: VariableWidthMLPForClassification,
    fit_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device | str,
    config: OTConfig,
) -> dict[str, object]:
    """Run OT, GW, or FGW end to end on shared pair-bank splits."""
    device = torch.device(device)
    sites = enumerate_canonical_sites(model, resolution=config.resolution)
    if config.selection_verbose:
        print(
            f"{str(config.method).upper()} training "
            f"| fit_examples={fit_bank.size} "
            f"| calibration_examples={calibration_bank.size} "
            f"| holdout_examples={holdout_bank.size} "
            f"| sites={len(sites)} "
            f"| target_vars={len(config.target_vars)}"
        )
    base_logits = collect_base_logits(model, fit_bank.base_inputs, config.batch_size, device)
    variable_signatures = build_variable_signatures(
        fit_bank,
        base_logits.shape[-1],
        config.target_vars,
    )
    site_signatures = collect_site_signatures(
        model=model,
        bank=fit_bank,
        sites=sites,
        base_logits=base_logits,
        batch_size=config.batch_size,
        device=device,
    )

    p = np.ones(variable_signatures.shape[0], dtype=np.float64) / variable_signatures.shape[0]
    q = np.ones(site_signatures.shape[0], dtype=np.float64) / site_signatures.shape[0]

    if config.method == "gw":
        cost_var, cost_site = build_geometry_costs(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        transport, transport_meta = solve_gw_transport(cost_var, cost_site, p, q, config)
        extra_payload = {"cost_var": cost_var.tolist(), "cost_site": cost_site.tolist()}
    elif config.method == "ot":
        cost_cross = build_cross_cost(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        transport, transport_meta = solve_ot_transport(cost_cross, p, q, config)
        extra_payload = {"cost_cross": cost_cross.tolist()}
    elif config.method == "fgw":
        cost_var, cost_site = build_geometry_costs(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        cost_cross = build_cross_cost(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        transport, transport_meta = solve_fgw_transport(cost_cross, cost_var, cost_site, p, q, config)
        extra_payload = {
            "cost_cross": cost_cross.tolist(),
            "cost_var": cost_var.tolist(),
            "cost_site": cost_site.tolist(),
        }
    else:
        raise ValueError(f"Unsupported alignment method {config.method}")

    if config.selection_verbose:
        print(
            f"{str(config.method).upper()} training complete "
            f"| transport_shape={tuple(int(dim) for dim in transport.shape)}"
        )

    rankings = build_rankings(transport, sites, config.target_vars, config.ranking_k)
    normalized_transport = normalize_transport_rows(transport)
    selection_payload = select_transport_hyperparameters(
        method_name=config.method,
        model=model,
        calibration_bank=calibration_bank,
        sites=sites,
        normalized_transport=normalized_transport,
        rankings=rankings,
        target_vars=config.target_vars,
        batch_size=config.batch_size,
        device=device,
        config=config,
    )
    selected_top_k_by_variable = {
        variable: int(selection_payload["selected_top_k_by_variable"][variable])
        for variable in config.target_vars
    }
    selected_lambda_by_variable = {
        variable: float(selection_payload["selected_lambda_by_variable"][variable])
        for variable in config.target_vars
    }
    selected_transport = truncate_transport_rows(
        normalized_transport,
        [selected_top_k_by_variable[variable] for variable in config.target_vars],
        renormalize=True,
    )
    calibration_results_by_variable = {
        str(record["variable"]): record for record in selection_payload["results"]
    }
    results = []
    layer_masks_by_variable = {}
    for variable_index, variable in enumerate(config.target_vars):
        variable_results, variable_layer_masks = evaluate_soft_transport_interventions(
            method_name=config.method,
            model=model,
            evaluation_bank=holdout_bank,
            sites=sites,
            transport_weights=selected_transport[variable_index : variable_index + 1],
            rankings={str(variable): rankings[str(variable)]},
            target_vars=(str(variable),),
            top_k_by_variable={str(variable): selected_top_k_by_variable[str(variable)]},
            lambda_by_variable={str(variable): selected_lambda_by_variable[str(variable)]},
            batch_size=config.batch_size,
            device=device,
        )
        layer_masks_by_variable.update(variable_layer_masks)
        record = variable_results[0]
        calibration_record = calibration_results_by_variable.get(str(record["variable"]), {})
        record["selection_exact_acc"] = float(calibration_record.get("exact_acc", 0.0))
        record["selection_mean_shared_digits"] = float(
            calibration_record.get("mean_shared_digits", 0.0)
        )
        record["calibration_exact_acc"] = float(calibration_record.get("exact_acc", 0.0))
        record["calibration_mean_shared_digits"] = float(
            calibration_record.get("mean_shared_digits", 0.0)
        )
        if config.selection_verbose:
            print(
                f"{str(config.method).upper()} [{record['variable']}] selected {record['site_label']} "
                f"| calibration_exact={float(record['calibration_exact_acc']):.4f} "
                f"| calibration_shared={float(record['calibration_mean_shared_digits']):.4f} "
                f"| holdout_exact={float(record['exact_acc']):.4f} "
                f"| holdout_shared={float(record['mean_shared_digits']):.4f}"
            )
        results.append(record)

    return {
        "method": config.method,
        "transport_meta": transport_meta,
        "train_bank": fit_bank.metadata(),
        "calibration_bank": calibration_bank.metadata(),
        "test_bank": holdout_bank.metadata(),
        "target_vars": list(config.target_vars),
        "resolution": config.resolution,
        "site_labels": [site.label for site in sites],
        "site_dims": [list(site.dims) for site in sites],
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "selected_transport": selected_transport.tolist(),
        "rankings": rankings,
        "selected_hyperparameters": {
            "top_k_by_variable": selected_top_k_by_variable,
            "lambda_by_variable": selected_lambda_by_variable,
            "average_calibration_exact_acc": float(selection_payload["average_calibration_exact_acc"]),
            "average_calibration_mean_shared_digits": float(selection_payload["average_calibration_mean_shared_digits"]),
        },
        "calibration_sweep": selection_payload["search_records"],
        "layer_masks_by_variable": {
            variable: {str(layer): mask.tolist() for layer, mask in layer_masks.items()}
            for variable, layer_masks in layer_masks_by_variable.items()
        },
        "results": results,
        **extra_payload,
    }
