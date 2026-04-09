"""Transport-based alignment and intervention for MCQA residual-stream sites."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cdist

from equality_experiment.ot import (
    _squared_euclidean_cost,
    _transport_validation_stats,
    _is_valid_balanced_transport,
    sinkhorn_uniform_ot,
    sinkhorn_unbalanced_ot,
)

from . import _env  # noqa: F401
from .data import MCQAPairBank
from .intervention import run_soft_residual_intervention
from .metrics import metrics_from_logits
from .signatures import collect_base_logits, collect_site_signatures
from .sites import ResidualSite


@dataclass(frozen=True)
class OTConfig:
    """Hyperparameters for OT/UOT alignment and intervention runs."""

    method: str = "ot"
    batch_size: int = 16
    epsilon: float = 1.0
    tau: float = 1.0
    uot_beta_abstract: float = 1.0
    uot_beta_neural: float = 1.0
    max_iter: int = 500
    tol: float = 1e-9
    signature_mode: str = "answer_logit_delta"
    top_k_values: tuple[int, ...] | None = None
    lambda_values: tuple[float, ...] = (1.0,)
    selection_verbose: bool = True


def build_rankings(transport: np.ndarray, sites: list[ResidualSite], ranking_k: int) -> list[dict[str, object]]:
    order = np.argsort(-transport[0], kind="stable")[: int(ranking_k)]
    return [
        {
            "site_index": int(site_index),
            "site_label": sites[int(site_index)].label,
            "layer": int(sites[int(site_index)].layer),
            "token_position_id": str(sites[int(site_index)].token_position_id),
            "dim_start": int(sites[int(site_index)].dim_start),
            "dim_end": int(sites[int(site_index)].dim_end),
            "transport_mass": float(transport[0, int(site_index)]),
        }
        for site_index in order
    ]


def normalize_transport_rows(transport: np.ndarray) -> np.ndarray:
    row_sums = transport.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    return transport / safe_row_sums


def truncate_transport_rows(normalized_transport: np.ndarray, top_k: int, renormalize: bool = False) -> np.ndarray:
    truncated = np.zeros_like(normalized_transport)
    limit = max(1, min(int(top_k), normalized_transport.shape[1]))
    order = np.argsort(-normalized_transport[0], kind="stable")[:limit]
    truncated[0, order] = normalized_transport[0, order]
    if renormalize:
        row_sum = float(truncated[0].sum())
        if row_sum > 0.0:
            truncated[0] = truncated[0] / row_sum
    return truncated


def solve_ot_transport(variable_signature: torch.Tensor, site_signatures: torch.Tensor, config: OTConfig) -> tuple[np.ndarray, dict[str, object]]:
    variable_signature = variable_signature.reshape(1, -1)
    site_signatures = site_signatures.reshape(site_signatures.shape[0], -1)
    cost_cross = _squared_euclidean_cost(variable_signature, site_signatures).detach().cpu().numpy()
    p = np.ones(variable_signature.shape[0], dtype=np.float64) / float(variable_signature.shape[0])
    q = np.ones(site_signatures.shape[0], dtype=np.float64) / float(site_signatures.shape[0])
    temperature = float(config.tau)
    regularization = float(config.epsilon) * temperature
    transport_tensor, transport_cost = sinkhorn_uniform_ot(
        variable_signature,
        site_signatures,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        temperature=temperature,
        tol=float(config.tol),
    )
    transport = transport_tensor.detach().cpu().numpy()
    meta = {
        "method": "ot",
        "regularization_used": regularization,
        "tau_used": temperature,
        "epsilon_config": float(config.epsilon),
        "transport_cost": float(transport_cost),
        **_transport_validation_stats(transport, p, q),
    }
    if _is_valid_balanced_transport(transport, p, q, float(config.tol)):
        return transport, meta
    meta.update({"failed": True, "failure_reason": "invalid_balanced_transport"})
    return transport, meta


def solve_uot_transport(variable_signature: torch.Tensor, site_signatures: torch.Tensor, config: OTConfig) -> tuple[np.ndarray, dict[str, object]]:
    variable_signature = variable_signature.reshape(1, -1)
    site_signatures = site_signatures.reshape(site_signatures.shape[0], -1)
    temperature = float(config.tau)
    regularization = float(config.epsilon) * temperature
    transport_tensor, info = sinkhorn_unbalanced_ot(
        variable_signature,
        site_signatures,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        temperature=temperature,
        tau_abstract=float(config.uot_beta_abstract),
        tau_neural=float(config.uot_beta_neural),
    )
    transport = transport_tensor.detach().cpu().numpy()
    meta = {
        "method": "uot",
        "regularization_used": regularization,
        "tau_used": temperature,
        "uot_beta_abstract": float(config.uot_beta_abstract),
        "uot_beta_neural": float(config.uot_beta_neural),
        "epsilon_config": float(config.epsilon),
        **info,
    }
    if np.isfinite(transport).all() and float(np.sum(transport)) > 0.0:
        return transport, meta
    meta.update({"failed": True, "failure_reason": "invalid_unbalanced_transport"})
    return transport, meta


def _evaluate_soft_intervention(
    *,
    model,
    bank: MCQAPairBank,
    sites: list[ResidualSite],
    selected_transport: np.ndarray,
    top_k: int,
    strength: float,
    batch_size: int,
    device: torch.device,
    tokenizer,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    site_weights = {
        sites[index]: float(selected_transport[0, index])
        for index in range(selected_transport.shape[1])
        if float(selected_transport[0, index]) > 0.0
    }
    logits_chunks = []
    for start in range(0, bank.size, batch_size):
        end = min(start + batch_size, bank.size)
        logits = run_soft_residual_intervention(
            model=model,
            base_input_ids=bank.base_input_ids[start:end].to(device),
            base_attention_mask=bank.base_attention_mask[start:end].to(device),
            source_input_ids=bank.source_input_ids[start:end].to(device),
            source_attention_mask=bank.source_attention_mask[start:end].to(device),
            site_weights=site_weights,
            strength=strength,
            base_position_by_id={
                key: value[start:end] for key, value in bank.base_position_by_id.items()
            },
            source_position_by_id={
                key: value[start:end] for key, value in bank.source_position_by_id.items()
            },
        )
        logits_chunks.append(logits.detach().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    ranking = build_rankings(selected_transport, sites, ranking_k=max(1, top_k))
    record = {
        "method": "soft_transport",
        "variable": bank.target_var,
        "split": bank.split,
        "site_label": f"soft:k{int(top_k)},l{float(strength):g}",
        "top_k": int(top_k),
        "lambda": float(strength),
        "top_site_label": ranking[0]["site_label"] if ranking else None,
        "selected_site_labels": [site.label for site in site_weights],
        **metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    return record, ranking


def _select_hyperparameters(
    *,
    model,
    calibration_bank: MCQAPairBank,
    sites: list[ResidualSite],
    normalized_transport: np.ndarray,
    batch_size: int,
    device: torch.device,
    tokenizer,
    config: OTConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    top_k_values = tuple(range(1, normalized_transport.shape[1] + 1)) if config.top_k_values is None else tuple(config.top_k_values)
    best = None
    sweep_records: list[dict[str, object]] = []
    for top_k in top_k_values:
        truncated = truncate_transport_rows(normalized_transport, int(top_k), renormalize=True)
        for strength in config.lambda_values:
            result, ranking = _evaluate_soft_intervention(
                model=model,
                bank=calibration_bank,
                sites=sites,
                selected_transport=truncated,
                top_k=int(top_k),
                strength=float(strength),
                batch_size=batch_size,
                device=device,
                tokenizer=tokenizer,
            )
            candidate = {
                "top_k": int(top_k),
                "lambda": float(strength),
                "result": result,
                "ranking": ranking,
                "exact_acc": float(result["exact_acc"]),
            }
            sweep_records.append(candidate)
            if best is None or float(candidate["exact_acc"]) > float(best["exact_acc"]):
                best = candidate
    if best is None:
        raise RuntimeError(f"Failed to select OT/UOT hyperparameters for {calibration_bank.target_var}")
    return best, sweep_records


def run_alignment_pipeline(
    *,
    model,
    fit_bank: MCQAPairBank,
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[ResidualSite],
    device: torch.device | str,
    tokenizer,
    config: OTConfig,
) -> dict[str, object]:
    """Run OT or UOT for one MCQA target variable."""
    device = torch.device(device)
    base_logits = collect_base_logits(model=model, bank=fit_bank, batch_size=config.batch_size, device=device)
    site_signatures = collect_site_signatures(
        model=model,
        bank=fit_bank,
        sites=sites,
        base_logits=base_logits,
        batch_size=config.batch_size,
        device=device,
        signature_mode=config.signature_mode,
    )
    from .metrics import build_variable_signature

    variable_signature = build_variable_signature(fit_bank, config.signature_mode)
    if config.method == "ot":
        transport, transport_meta = solve_ot_transport(variable_signature, site_signatures, config)
    elif config.method == "uot":
        transport, transport_meta = solve_uot_transport(variable_signature, site_signatures, config)
    else:
        raise ValueError(f"Unsupported MCQA transport method {config.method}")
    normalized_transport = normalize_transport_rows(transport)
    selected, calibration_sweep = _select_hyperparameters(
        model=model,
        calibration_bank=calibration_bank,
        sites=sites,
        normalized_transport=normalized_transport,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        config=config,
    )
    top_k = int(selected["top_k"])
    strength = float(selected["lambda"])
    selected_transport = truncate_transport_rows(normalized_transport, top_k, renormalize=True)
    holdout_result, holdout_ranking = _evaluate_soft_intervention(
        model=model,
        bank=holdout_bank,
        sites=sites,
        selected_transport=selected_transport,
        top_k=top_k,
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
    )
    holdout_result["method"] = config.method
    holdout_result["selection_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["calibration_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["signature_mode"] = str(config.signature_mode)
    holdout_result["selected_transport_nonzero"] = int((selected_transport[0] > 0.0).sum())
    return {
        "target_var": fit_bank.target_var,
        "signature_mode": config.signature_mode,
        "transport": transport.tolist(),
        "normalized_transport": normalized_transport.tolist(),
        "selected_transport": selected_transport.tolist(),
        "transport_meta": transport_meta,
        "selected_hyperparameters": {
            "top_k": top_k,
            "lambda": strength,
            "signature_mode": config.signature_mode,
        },
        "ranking": holdout_ranking,
        "calibration_sweep": calibration_sweep,
        "results": [holdout_result],
    }
