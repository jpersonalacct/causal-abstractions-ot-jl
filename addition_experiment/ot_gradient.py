"""Gradient-based single-layer transport policy search for OT, GW, and FGW."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback only when tqdm is unavailable
    tqdm = None

from variable_width_mlp import VariableWidthMLPForClassification

from .metrics import metrics_from_logits
from .ot import (
    OTConfig,
    _format_hparam_value,
    build_cross_cost,
    build_geometry_costs,
    build_rankings,
    build_variable_signatures,
    collect_base_logits,
    collect_site_signatures,
    evaluate_soft_transport_interventions,
    normalize_transport_rows,
    solve_fgw_transport,
    solve_gw_transport,
    solve_ot_transport,
)
from .pair_bank import PairBank
from .pyvene_utils import CanonicalSite, enumerate_canonical_sites


@dataclass(frozen=True)
class OTGradientConfig:
    """Hyperparameters for gradient-based single-layer transport policy search."""

    method: str = "ot"
    batch_size: int = 128
    geometry_metric: str = "cosine"
    normalize_cost_matrices: bool = True
    epsilon: float = 5e-2
    max_iter: int = 500
    tol: float = 1e-9
    verbose: bool = False
    epsilon_retry_multipliers: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0, 100.0)
    ranking_k: int = 5
    resolution: int = 1
    alpha: float = 0.5
    target_vars: tuple[str, ...] = ("S1", "C1", "S2", "C2")
    policy_learning_rate: float = 5e-2
    policy_epochs: int = 1000
    policy_min_epochs: int = 25
    policy_plateau_patience: int = 5
    policy_plateau_rel_delta: float = 5e-3
    policy_temperature: float = 1.0
    policy_eval_interval: int = 1
    fixed_top_k: int | None = None
    fixed_lambda: float | None = None
    selection_verbose: bool = True


def _inverse_softplus(value: float) -> float:
    """Return a stable inverse-softplus initialization value."""
    value = float(max(value, 1e-6))
    return float(np.log(np.expm1(value)))


def ranked_cutoff_gates(
    cutoff: torch.Tensor,
    num_sites: int,
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    """Return a soft top-k gate that decreases with site rank."""
    indices = torch.arange(1, num_sites + 1, device=device, dtype=torch.float32)
    return torch.sigmoid((cutoff + 0.5 - indices) / float(temperature))


def continuous_cutoff_to_top_k(cutoff_value: float, num_sites: int) -> int:
    """Round a learned continuous cutoff to a valid integer top-k."""
    if num_sites <= 0:
        raise ValueError("num_sites must be positive")
    return max(1, min(int(round(float(cutoff_value))), int(num_sites)))


def build_site_projection_matrix(
    sites: list[CanonicalSite],
    layer_width: int,
    device: torch.device,
) -> torch.Tensor:
    """Map site weights to neuron mask weights within one layer."""
    projection = torch.zeros((len(sites), layer_width), device=device, dtype=torch.float32)
    for site_index, site in enumerate(sites):
        per_dim_weight = 1.0 / float(len(site.dims))
        for dim in site.dims:
            projection[site_index, int(dim)] = per_dim_weight
    return projection


def build_layer_mask_from_site_weights(
    site_weights: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    """Collapse site weights into a neuron mask for one layer."""
    return torch.matmul(site_weights.view(1, -1), projection).view(-1)


def run_single_layer_soft_intervention_logits(
    model: VariableWidthMLPForClassification,
    base_inputs: torch.Tensor,
    source_inputs: torch.Tensor,
    layer: int,
    layer_mask: torch.Tensor,
    strength: torch.Tensor,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Apply one differentiable single-layer intervention and return logits."""
    outputs = []
    device = torch.device(device)
    mask = layer_mask.to(device=device, dtype=torch.float32).view(1, 1, -1)
    scalar_strength = strength.to(device=device, dtype=torch.float32).view(1, 1, 1)

    model.eval()
    for start in range(0, base_inputs.shape[0], batch_size):
        end = min(start + batch_size, base_inputs.shape[0])
        base_hidden = base_inputs[start:end].to(device)
        source_hidden = source_inputs[start:end].to(device)
        if base_hidden.ndim == 2:
            base_hidden = base_hidden.unsqueeze(1)
        if source_hidden.ndim == 2:
            source_hidden = source_hidden.unsqueeze(1)

        for current_layer, block in enumerate(model.h):
            source_hidden = block(source_hidden)
            base_hidden = block(base_hidden)
            if current_layer == int(layer):
                base_hidden = base_hidden + scalar_strength * mask * (source_hidden - base_hidden)

        logits = model.score(base_hidden)
        if model.config.squeeze_output:
            logits = logits.squeeze(1)
        outputs.append(logits)
    return torch.cat(outputs, dim=0)


def evaluate_single_layer_soft_policy(
    model: VariableWidthMLPForClassification,
    bank: PairBank,
    variable: str,
    layer: int,
    projection: torch.Tensor,
    sorted_transport_weights: torch.Tensor,
    raw_cutoff: torch.Tensor | None,
    raw_lambda: torch.Tensor | None,
    batch_size: int,
    device: torch.device,
    temperature: float,
    fixed_top_k: int | None,
    fixed_lambda: float | None,
) -> dict[str, object]:
    """Evaluate the current soft policy state on one bank split."""
    def _as_float(value: torch.Tensor | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    num_sites = int(sorted_transport_weights.numel())
    if fixed_top_k is None:
        if raw_cutoff is None:
            raise ValueError("raw_cutoff must be provided when fixed_top_k is None")
        cutoff = float(num_sites) * torch.sigmoid(raw_cutoff)
        gates = ranked_cutoff_gates(cutoff, num_sites, temperature, device)
    else:
        cutoff = float(max(1, min(int(fixed_top_k), int(num_sites))))
        gates = torch.zeros(num_sites, device=device, dtype=torch.float32)
        gates[: int(cutoff)] = 1.0
    gated_weights = gates * sorted_transport_weights
    gated_sum = gated_weights.sum()
    if float(gated_sum.detach().cpu()) <= 0.0:
        normalized_weights = torch.full_like(gated_weights, 1.0 / float(num_sites))
    else:
        normalized_weights = gated_weights / gated_sum
    layer_mask = build_layer_mask_from_site_weights(normalized_weights, projection)
    if fixed_lambda is None:
        if raw_lambda is None:
            raise ValueError("raw_lambda must be provided when fixed_lambda is None")
        strength = F.softplus(raw_lambda)
    else:
        strength = torch.tensor(float(fixed_lambda), device=device, dtype=torch.float32)
    logits = run_single_layer_soft_intervention_logits(
        model=model,
        base_inputs=bank.base_inputs,
        source_inputs=bank.source_inputs,
        layer=layer,
        layer_mask=layer_mask,
        strength=strength,
        batch_size=batch_size,
        device=device,
    )
    labels = bank.cf_labels_by_var[variable].to(device=logits.device, dtype=torch.long).view(-1)
    cross_entropy = F.cross_entropy(logits, labels)
    metrics = metrics_from_logits(logits.detach().cpu(), bank.cf_labels_by_var[variable])
    return {
        "cross_entropy": float(cross_entropy.detach().cpu()),
        "exact_acc": float(metrics["exact_acc"]),
        "mean_shared_digits": float(metrics["mean_shared_digits"]),
        "cutoff": _as_float(cutoff),
        "lambda": _as_float(strength),
        "layer_mask": layer_mask.detach().cpu(),
    }


def optimize_layer_policy(
    model: VariableWidthMLPForClassification,
    calibration_bank: PairBank,
    variable: str,
    layer: int,
    layer_sites: list[CanonicalSite],
    sorted_transport_weights: torch.Tensor,
    batch_size: int,
    device: torch.device,
    config: OTGradientConfig,
) -> dict[str, object]:
    """Learn a soft cutoff and lambda for one variable-layer pair."""
    projection = build_site_projection_matrix(
        layer_sites,
        layer_width=int(model.config.hidden_dims[layer]),
        device=device,
    )
    dataset = TensorDataset(
        calibration_bank.base_inputs,
        calibration_bank.source_inputs,
        calibration_bank.cf_labels_by_var[variable],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    raw_cutoff = None
    raw_lambda = None
    optimizer_parameters = []
    if config.fixed_top_k is None:
        raw_cutoff = torch.nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float32))
        optimizer_parameters.append(raw_cutoff)
    if config.fixed_lambda is None:
        raw_lambda = torch.nn.Parameter(
            torch.tensor(_inverse_softplus(1.0), device=device, dtype=torch.float32)
        )
        optimizer_parameters.append(raw_lambda)
    optimizer = None
    if optimizer_parameters:
        optimizer = torch.optim.Adam(optimizer_parameters, lr=float(config.policy_learning_rate))

    best_epoch_summary = None
    loss_history = []
    num_sites = int(sorted_transport_weights.numel())
    transport_weights_device = sorted_transport_weights.to(device=device, dtype=torch.float32)
    best_selection_cross_entropy = None
    plateau_steps = 0
    stopping_reason = None
    progress_bar = None
    if config.selection_verbose and tqdm is not None:
        progress_bar = tqdm(
            total=int(config.policy_epochs),
            desc=f"{str(config.method).upper()} [{variable}] L{layer}",
            unit="epoch",
            dynamic_ncols=True,
            leave=True,
        )

    if optimizer is None:
        best_epoch_summary = evaluate_single_layer_soft_policy(
            model=model,
            bank=calibration_bank,
            variable=variable,
            layer=layer,
            projection=projection,
            sorted_transport_weights=transport_weights_device,
            raw_cutoff=raw_cutoff,
            raw_lambda=raw_lambda,
            batch_size=batch_size,
            device=device,
            temperature=config.policy_temperature,
            fixed_top_k=config.fixed_top_k,
            fixed_lambda=config.fixed_lambda,
        )
        best_epoch_summary["epoch"] = 0
        best_epoch_summary["train_cross_entropy"] = float(best_epoch_summary["cross_entropy"])
        if progress_bar is not None:
            progress_bar.update(int(config.policy_epochs))
            progress_bar.set_postfix_str(
                (
                    f"fixed_top_k={int(config.fixed_top_k)}"
                    if config.fixed_top_k is not None
                    else "fixed_policy=1"
                ),
                refresh=False,
            )
            progress_bar.close()
        return {
            "variable": variable,
            "layer": int(layer),
            "num_sites": num_sites,
            "continuous_cutoff": float(best_epoch_summary["cutoff"]),
            "lambda": float(best_epoch_summary["lambda"]),
            "selection_cross_entropy": float(best_epoch_summary["cross_entropy"]),
            "selection_exact_acc": float(best_epoch_summary["exact_acc"]),
            "selection_mean_shared_digits": float(best_epoch_summary["mean_shared_digits"]),
            "fixed_top_k": None if config.fixed_top_k is None else int(config.fixed_top_k),
            "fixed_lambda": None if config.fixed_lambda is None else float(config.fixed_lambda),
            "epoch": int(best_epoch_summary["epoch"]),
            "epochs_ran": 0,
            "stopped_early": False,
            "stopping_reason": "fixed_policy_no_optimization",
            "train_cross_entropy_history": loss_history,
        }

    try:
        for epoch_index in range(int(config.policy_epochs)):
            batch_losses = []
            for batch_base, batch_source, batch_labels in loader:
                if config.fixed_top_k is None:
                    if raw_cutoff is None:
                        raise ValueError("raw_cutoff unexpectedly missing during optimization")
                    cutoff = float(num_sites) * torch.sigmoid(raw_cutoff)
                    gates = ranked_cutoff_gates(cutoff, num_sites, config.policy_temperature, device)
                else:
                    cutoff = float(max(1, min(int(config.fixed_top_k), int(num_sites))))
                    gates = torch.zeros(num_sites, device=device, dtype=torch.float32)
                    gates[: int(cutoff)] = 1.0
                gated_weights = gates * transport_weights_device
                gated_sum = gated_weights.sum()
                if torch.isclose(gated_sum.detach(), torch.tensor(0.0, device=device)):
                    normalized_weights = torch.full_like(gated_weights, 1.0 / float(num_sites))
                else:
                    normalized_weights = gated_weights / gated_sum
                layer_mask = build_layer_mask_from_site_weights(normalized_weights, projection)
                if config.fixed_lambda is None:
                    if raw_lambda is None:
                        raise ValueError("raw_lambda unexpectedly missing during optimization")
                    strength = F.softplus(raw_lambda)
                else:
                    strength = torch.tensor(float(config.fixed_lambda), device=device, dtype=torch.float32)
                logits = run_single_layer_soft_intervention_logits(
                    model=model,
                    base_inputs=batch_base,
                    source_inputs=batch_source,
                    layer=layer,
                    layer_mask=layer_mask,
                    strength=strength,
                    batch_size=batch_size,
                    device=device,
                )
                loss = F.cross_entropy(logits, batch_labels.to(device=device, dtype=torch.long).view(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            epoch_loss = sum(batch_losses) / max(len(batch_losses), 1)
            loss_history.append(epoch_loss)

            status_bits = [f"train_ce={epoch_loss:.4f}"]
            if (epoch_index + 1) % int(config.policy_eval_interval) == 0:
                epoch_summary = evaluate_single_layer_soft_policy(
                    model=model,
                    bank=calibration_bank,
                    variable=variable,
                    layer=layer,
                    projection=projection,
                    sorted_transport_weights=transport_weights_device,
                    raw_cutoff=raw_cutoff,
                    raw_lambda=raw_lambda,
                    batch_size=batch_size,
                    device=device,
                    temperature=config.policy_temperature,
                    fixed_top_k=config.fixed_top_k,
                    fixed_lambda=config.fixed_lambda,
                )
                epoch_summary["epoch"] = int(epoch_index + 1)
                epoch_summary["train_cross_entropy"] = float(epoch_loss)
                status_bits.append(f"cal_ce={float(epoch_summary['cross_entropy']):.4f}")
                status_bits.append(f"cutoff={float(epoch_summary['cutoff']):.2f}")
                status_bits.append(f"lambda={float(epoch_summary['lambda']):.3f}")
                if best_epoch_summary is None:
                    best_epoch_summary = epoch_summary
                    best_selection_cross_entropy = float(epoch_summary["cross_entropy"])
                else:
                    current_key = (
                        -float(epoch_summary["cross_entropy"]),
                        float(epoch_summary["exact_acc"]),
                        float(epoch_summary["mean_shared_digits"]),
                    )
                    best_key = (
                        -float(best_epoch_summary["cross_entropy"]),
                        float(best_epoch_summary["exact_acc"]),
                        float(best_epoch_summary["mean_shared_digits"]),
                    )
                    if current_key > best_key:
                        best_epoch_summary = epoch_summary
                    current_cross_entropy = float(epoch_summary["cross_entropy"])
                    relative_threshold = float(best_selection_cross_entropy) * (
                        1.0 - float(config.policy_plateau_rel_delta)
                    )
                    improved = current_cross_entropy < relative_threshold
                    if improved:
                        best_selection_cross_entropy = current_cross_entropy
                        plateau_steps = 0
                    else:
                        plateau_steps += 1
                    if (
                        int(config.policy_plateau_patience) > 0
                        and epoch_index + 1 >= int(config.policy_min_epochs)
                        and plateau_steps >= int(config.policy_plateau_patience)
                    ):
                        stopping_reason = (
                            "calibration_cross_entropy_plateau "
                            f"for {int(config.policy_plateau_patience)} evals"
                        )
                        if progress_bar is not None:
                            progress_bar.set_postfix_str(
                                " | ".join([*status_bits, "early_stop=1"]),
                                refresh=False,
                            )
                        break
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix_str(" | ".join(status_bits), refresh=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    if best_epoch_summary is None:
        best_epoch_summary = evaluate_single_layer_soft_policy(
            model=model,
            bank=calibration_bank,
            variable=variable,
            layer=layer,
            projection=projection,
            sorted_transport_weights=transport_weights_device,
            raw_cutoff=raw_cutoff,
            raw_lambda=raw_lambda,
            batch_size=batch_size,
            device=device,
            temperature=config.policy_temperature,
            fixed_top_k=config.fixed_top_k,
            fixed_lambda=config.fixed_lambda,
        )
        best_epoch_summary["epoch"] = int(config.policy_epochs)
        best_epoch_summary["train_cross_entropy"] = float(loss_history[-1]) if loss_history else 0.0

    return {
        "variable": variable,
        "layer": int(layer),
        "num_sites": num_sites,
        "continuous_cutoff": float(best_epoch_summary["cutoff"]),
        "lambda": float(best_epoch_summary["lambda"]),
        "selection_cross_entropy": float(best_epoch_summary["cross_entropy"]),
        "selection_exact_acc": float(best_epoch_summary["exact_acc"]),
        "selection_mean_shared_digits": float(best_epoch_summary["mean_shared_digits"]),
        "fixed_top_k": None if config.fixed_top_k is None else int(config.fixed_top_k),
        "fixed_lambda": None if config.fixed_lambda is None else float(config.fixed_lambda),
        "epoch": int(best_epoch_summary["epoch"]),
        "epochs_ran": len(loss_history),
        "stopped_early": stopping_reason is not None,
        "stopping_reason": stopping_reason,
        "train_cross_entropy_history": loss_history,
    }


def build_single_layer_selected_transport(
    normalized_transport: np.ndarray,
    row_index: int,
    layer_site_indices: list[int],
    top_k: int,
) -> np.ndarray:
    """Build a full transport row that keeps only the top-k sites within one layer."""
    selected_transport = np.zeros_like(normalized_transport[row_index])
    layer_weights = normalized_transport[row_index, layer_site_indices]
    order = np.argsort(-layer_weights, kind="stable")[: int(top_k)]
    selected_indices = [int(layer_site_indices[index]) for index in order]
    selected_transport[selected_indices] = normalized_transport[row_index, selected_indices]
    row_sum = float(selected_transport.sum())
    if row_sum > 0.0:
        selected_transport = selected_transport / row_sum
    return selected_transport


def run_alignment_gradient_pipeline(
    model: VariableWidthMLPForClassification,
    fit_bank: PairBank,
    calibration_bank: PairBank,
    holdout_bank: PairBank,
    device: torch.device | str,
    config: OTGradientConfig,
) -> dict[str, object]:
    """Run OT, GW, or FGW with gradient-based single-layer policy search."""
    device = torch.device(device)
    sites = enumerate_canonical_sites(model, resolution=config.resolution)
    if config.selection_verbose:
        print(
            f"{str(config.method).upper()} gradient training "
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

    ot_solver_config = OTConfig(
        method=config.method,
        batch_size=config.batch_size,
        geometry_metric=config.geometry_metric,
        normalize_cost_matrices=config.normalize_cost_matrices,
        epsilon=config.epsilon,
        max_iter=config.max_iter,
        tol=config.tol,
        verbose=config.verbose,
        epsilon_retry_multipliers=config.epsilon_retry_multipliers,
        ranking_k=config.ranking_k,
        resolution=config.resolution,
        alpha=config.alpha,
        target_vars=config.target_vars,
        selection_verbose=False,
    )

    extra_payload = {}
    if config.method == "gw":
        cost_var, cost_site = build_geometry_costs(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        transport, transport_meta = solve_gw_transport(cost_var, cost_site, p, q, ot_solver_config)
        extra_payload = {"cost_var": cost_var.tolist(), "cost_site": cost_site.tolist()}
    elif config.method == "ot":
        cost_cross = build_cross_cost(
            variable_signatures,
            site_signatures,
            metric=config.geometry_metric,
            normalize=config.normalize_cost_matrices,
        )
        transport, transport_meta = solve_ot_transport(cost_cross, p, q, ot_solver_config)
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
        transport, transport_meta = solve_fgw_transport(cost_cross, cost_var, cost_site, p, q, ot_solver_config)
        extra_payload = {
            "cost_cross": cost_cross.tolist(),
            "cost_var": cost_var.tolist(),
            "cost_site": cost_site.tolist(),
        }
    else:
        raise ValueError(f"Unsupported alignment method {config.method}")

    rankings = build_rankings(transport, sites, config.target_vars, config.ranking_k)
    normalized_transport = normalize_transport_rows(transport)
    selected_rows = np.zeros_like(normalized_transport)
    selected_layer_by_variable = {}
    selected_top_k_by_variable = {}
    selected_lambda_by_variable = {}
    selected_cutoff_by_variable = {}
    layer_candidate_summaries = {}
    calibration_results = []
    holdout_results = []
    layer_masks_by_variable = {}

    for variable_index, variable in enumerate(config.target_vars):
        if config.selection_verbose:
            print(f"{str(config.method).upper()} [{variable}] gradient layer search")
        variable_candidates = []
        for layer in range(model.config.n_layer):
            layer_site_indices = [
                site_index for site_index, site in enumerate(sites) if int(site.layer) == int(layer)
            ]
            layer_sites = [sites[index] for index in layer_site_indices]
            sorted_pairs = sorted(
                (
                    (int(site_index), float(normalized_transport[variable_index, site_index]))
                    for site_index in layer_site_indices
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            sorted_indices = [site_index for site_index, _ in sorted_pairs]
            sorted_weights = torch.tensor(
                [weight for _, weight in sorted_pairs],
                dtype=torch.float32,
                device=device,
            )
            candidate = optimize_layer_policy(
                model=model,
                calibration_bank=calibration_bank,
                variable=str(variable),
                layer=int(layer),
                layer_sites=[sites[index] for index in sorted_indices],
                sorted_transport_weights=sorted_weights,
                batch_size=config.batch_size,
                device=device,
                config=config,
            )
            candidate["layer_site_indices"] = sorted_indices
            top_site_label = sites[sorted_indices[0]].label if sorted_indices else "n/a"
            candidate["top_site_label"] = top_site_label
            variable_candidates.append(candidate)
            if config.selection_verbose:
                print(
                    f"{str(config.method).upper()} [{variable}] layer={layer} "
                    f"| cutoff={candidate['continuous_cutoff']:.3f} "
                    f"| lambda={_format_hparam_value(candidate['lambda'])} "
                    f"| cal_ce={candidate['selection_cross_entropy']:.4f} "
                    f"| cal_exact={candidate['selection_exact_acc']:.4f} "
                    f"| cal_shared={candidate['selection_mean_shared_digits']:.4f}"
                )

        best_candidate = min(
            variable_candidates,
            key=lambda candidate: (
                float(candidate["selection_cross_entropy"]),
                -float(candidate["selection_exact_acc"]),
                -float(candidate["selection_mean_shared_digits"]),
            ),
        )
        selected_layer = int(best_candidate["layer"])
        selected_cutoff = float(best_candidate["continuous_cutoff"])
        if config.fixed_top_k is None:
            selected_top_k = continuous_cutoff_to_top_k(
                selected_cutoff,
                int(best_candidate["num_sites"]),
            )
        else:
            selected_top_k = max(1, min(int(config.fixed_top_k), int(best_candidate["num_sites"])))
        selected_lambda = float(best_candidate["lambda"])

        selected_layer_by_variable[str(variable)] = selected_layer
        selected_top_k_by_variable[str(variable)] = selected_top_k
        selected_lambda_by_variable[str(variable)] = selected_lambda
        selected_cutoff_by_variable[str(variable)] = selected_cutoff
        layer_candidate_summaries[str(variable)] = [
            {
                key: value
                for key, value in candidate.items()
                if key != "layer_site_indices"
            }
            for candidate in variable_candidates
        ]

        selected_row = build_single_layer_selected_transport(
            normalized_transport=normalized_transport,
            row_index=variable_index,
            layer_site_indices=list(best_candidate["layer_site_indices"]),
            top_k=selected_top_k,
        )
        selected_rows[variable_index] = selected_row
        layer_ranking = build_rankings(
            selected_row.reshape(1, -1),
            sites,
            (str(variable),),
            config.ranking_k,
        )
        calibration_record, calibration_layer_masks = evaluate_soft_transport_interventions(
            method_name=config.method,
            model=model,
            evaluation_bank=calibration_bank,
            sites=sites,
            transport_weights=selected_row.reshape(1, -1),
            rankings=layer_ranking,
            target_vars=(str(variable),),
            top_k_by_variable={str(variable): selected_top_k},
            lambda_by_variable={str(variable): selected_lambda},
            batch_size=config.batch_size,
            device=device,
        )
        holdout_record, holdout_layer_masks = evaluate_soft_transport_interventions(
            method_name=config.method,
            model=model,
            evaluation_bank=holdout_bank,
            sites=sites,
            transport_weights=selected_row.reshape(1, -1),
            rankings=layer_ranking,
            target_vars=(str(variable),),
            top_k_by_variable={str(variable): selected_top_k},
            lambda_by_variable={str(variable): selected_lambda},
            batch_size=config.batch_size,
            device=device,
        )
        calibration_result = calibration_record[0]
        holdout_result = holdout_record[0]
        holdout_result["site_label"] = (
            f"L{selected_layer}:soft:k{selected_top_k},l{_format_hparam_value(selected_lambda)}"
        )
        holdout_result["selected_layer"] = selected_layer
        holdout_result["continuous_cutoff"] = selected_cutoff
        holdout_result["selection_cross_entropy"] = float(best_candidate["selection_cross_entropy"])
        holdout_result["selection_exact_acc"] = float(calibration_result["exact_acc"])
        holdout_result["selection_mean_shared_digits"] = float(
            calibration_result["mean_shared_digits"]
        )
        holdout_result["calibration_exact_acc"] = float(calibration_result["exact_acc"])
        holdout_result["calibration_mean_shared_digits"] = float(
            calibration_result["mean_shared_digits"]
        )
        holdout_result["selection_objective"] = "calibration_cross_entropy"
        holdout_result["final_evaluation_policy"] = "hard_single_layer_top_k"
        calibration_results.append(calibration_result)
        holdout_results.append(holdout_result)
        layer_masks_by_variable.update(holdout_layer_masks)

        if config.selection_verbose:
            print(
                f"{str(config.method).upper()} [{variable}] selected "
                f"| layer={selected_layer} "
                f"| top_k={selected_top_k} "
                f"| lambda={_format_hparam_value(selected_lambda)} "
                f"| cal_ce={float(best_candidate['selection_cross_entropy']):.4f} "
                f"| cal_exact={float(calibration_result['exact_acc']):.4f} "
                f"| cal_shared={float(calibration_result['mean_shared_digits']):.4f} "
                f"| holdout_exact={float(holdout_result['exact_acc']):.4f} "
                f"| holdout_shared={float(holdout_result['mean_shared_digits']):.4f}"
            )

    return {
        "method": config.method,
        "transport_meta": transport_meta,
        "train_bank": fit_bank.metadata(),
        "calibration_bank": calibration_bank.metadata(),
        "test_bank": holdout_bank.metadata(),
        "target_vars": list(config.target_vars),
        "resolution": config.resolution,
        "selection_objective": "calibration_cross_entropy",
        "final_evaluation_policy": "hard_single_layer_top_k",
        "site_labels": [site.label for site in sites],
        "site_dims": [list(site.dims) for site in sites],
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "selected_transport": selected_rows.tolist(),
        "rankings": rankings,
        "selected_hyperparameters": {
            "selected_layer_by_variable": selected_layer_by_variable,
            "top_k_by_variable": selected_top_k_by_variable,
            "lambda_by_variable": selected_lambda_by_variable,
            "continuous_cutoff_by_variable": selected_cutoff_by_variable,
            "fixed_top_k": None if config.fixed_top_k is None else int(config.fixed_top_k),
            "fixed_lambda": None if config.fixed_lambda is None else float(config.fixed_lambda),
        },
        "layer_candidate_summaries": layer_candidate_summaries,
        "layer_masks_by_variable": {
            variable: {str(layer): mask.tolist() for layer, mask in layer_masks.items()}
            for variable, layer_masks in layer_masks_by_variable.items()
        },
        "calibration_results": calibration_results,
        "results": holdout_results,
        **extra_payload,
    }
