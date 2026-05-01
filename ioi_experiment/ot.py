"""Transport-based alignment and intervention for IOI residual-stream sites."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .data import IOIPairBank
from .intervention import run_soft_residual_intervention
from .metrics import das_metrics_from_logits, das_prediction_details_from_logits
from .signatures import build_variable_signature, collect_base_logits, collect_site_signatures
from .sites import ResidualSite


def _metrics_from_logits(
    logits: torch.Tensor, bank: IOIPairBank, tokenizer=None
) -> dict[str, object]:
    """IOI soft-intervention metrics with choice-constrained accuracy as the primary signal."""
    m = das_metrics_from_logits(logits, bank, tokenizer=tokenizer)
    # choice_acc (restricted to the two candidate names) is the semantically correct
    # primary metric for IOI; expose it as exact_acc so the calibration selector works
    # identically to the MCQA pipeline.
    m["exact_acc"] = m["choice_acc"]
    return m


def _prediction_details_from_logits(
    logits: torch.Tensor, bank: IOIPairBank, tokenizer=None
) -> dict[str, object]:
    return das_prediction_details_from_logits(logits, bank, tokenizer=tokenizer)


def _squared_euclidean_cost(u_points: torch.Tensor, v_points: torch.Tensor) -> torch.Tensor:
    """Compute squared Euclidean transport costs between two point clouds."""
    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    return torch.cdist(u, v, p=2).pow(2)


def _balanced_marginal_error(pi: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute the max absolute marginal violation for a balanced transport plan."""
    row_error = torch.max(torch.abs(pi.sum(dim=1) - a))
    col_error = torch.max(torch.abs(pi.sum(dim=0) - b))
    return float(torch.maximum(row_error, col_error).item())


def _transport_validation_stats(
    transport: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> dict[str, float]:
    """Summarize mass and marginal residuals for a candidate balanced transport."""
    transport = np.asarray(transport, dtype=float)
    row_residual = float(np.max(np.abs(transport.sum(axis=1) - p))) if transport.size else float("inf")
    col_residual = float(np.max(np.abs(transport.sum(axis=0) - q))) if transport.size else float("inf")
    total_mass = float(transport.sum())
    return {
        "matched_mass": total_mass,
        "max_row_residual": row_residual,
        "max_col_residual": col_residual,
    }


def _is_valid_balanced_transport(
    transport: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    tol: float,
) -> bool:
    """Return True only for finite, nonnegative, mass-preserving balanced couplings."""
    transport = np.asarray(transport, dtype=float)
    if transport.shape != (p.shape[0], q.shape[0]):
        return False
    if not np.isfinite(transport).all():
        return False
    if np.any(transport < 0.0):
        return False
    stats = _transport_validation_stats(transport, p, q)
    residual_tol = max(float(tol), 1e-6)
    mass_tol = max(float(tol), 1e-6)
    return (
        abs(stats["matched_mass"] - 1.0) <= mass_tol
        and stats["max_row_residual"] <= residual_tol
        and stats["max_col_residual"] <= residual_tol
    )


def sinkhorn_uniform_ot(
    u_points: torch.Tensor,
    v_points: torch.Tensor,
    epsilon: float,
    n_iter: int,
    temperature: float = 1.0,
    tol: float = 1e-9,
) -> tuple[torch.Tensor, float]:
    """Entropic OT with uniform marginals and squared Euclidean cost."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if tol < 0:
        raise ValueError("tol must be >= 0")

    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    m, n = u.size(0), v.size(0)

    a = torch.full((m,), 1.0 / m, dtype=torch.float32, device=u.device)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32, device=v.device)
    cost = _squared_euclidean_cost(u, v)
    kernel = torch.exp(-cost / (epsilon * temperature)).clamp_min(1e-30)

    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(n_iter):
        kr = kernel @ c
        r = a / kr.clamp_min(1e-30)
        kt = kernel.transpose(0, 1) @ r
        c = b / kt.clamp_min(1e-30)
        pi = r[:, None] * kernel * c[None, :]
        if _balanced_marginal_error(pi, a, b) <= float(tol):
            break

    pi = r[:, None] * kernel * c[None, :]
    ot_cost = float((pi * cost).sum().item())
    return pi, ot_cost


def sinkhorn_unbalanced_ot(
    u_points: torch.Tensor,
    v_points: torch.Tensor,
    epsilon: float,
    n_iter: int,
    temperature: float = 1.0,
    tau_abstract: float = 1.0e6,
    tau_neural: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Entropic unbalanced OT with KL penalties on marginals."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")
    if tau_abstract <= 0 or tau_neural <= 0:
        raise ValueError("tau_abstract and tau_neural must be > 0")

    u = u_points.to(dtype=torch.float32)
    v = v_points.to(dtype=torch.float32)
    m, n = u.size(0), v.size(0)

    a = torch.full((m,), 1.0 / m, dtype=torch.float32, device=u.device)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32, device=v.device)
    cost = _squared_euclidean_cost(u, v)
    kernel = torch.exp(-cost / (epsilon * temperature)).clamp_min(1e-30)

    rho_a = float(tau_abstract / (tau_abstract + epsilon))
    rho_b = float(tau_neural / (tau_neural + epsilon))

    r = torch.ones_like(a)
    c = torch.ones_like(b)
    for _ in range(n_iter):
        kr = kernel @ c
        r = (a / kr.clamp_min(1e-30)).pow(rho_a)
        kt = kernel.transpose(0, 1) @ r
        c = (b / kt.clamp_min(1e-30)).pow(rho_b)

    pi = r[:, None] * kernel * c[None, :]
    pi_row = pi.sum(dim=1)
    pi_col = pi.sum(dim=0)
    transport_cost = float((pi * cost).sum().item())
    kl_row = float(
        (
            pi_row * torch.log(pi_row.clamp_min(1e-30) / a.clamp_min(1e-30))
            - pi_row
            + a
        ).sum().item()
    )
    kl_col = float(
        (
            pi_col * torch.log(pi_col.clamp_min(1e-30) / b.clamp_min(1e-30))
            - pi_col
            + b
        ).sum().item()
    )
    total_obj = transport_cost + float(tau_abstract) * kl_row + float(tau_neural) * kl_col
    return pi, {
        "transport_cost": transport_cost,
        "kl_abstract": kl_row,
        "kl_neural": kl_col,
        "estimated_cost": total_obj,
        "matched_mass": float(pi.sum().item()),
    }


@dataclass(frozen=True)
class OTConfig:
    """Hyperparameters for OT/UOT alignment and intervention runs."""

    method: str = "ot"
    batch_size: int = 16
    epsilon: float = 1.0
    uot_beta_abstract: float = 1.0
    uot_beta_neural: float = 1.0
    max_iter: int = 500
    tol: float = 1e-9
    signature_mode: str = "answer_logit_delta"
    top_k_values: tuple[int, ...] | None = None
    lambda_values: tuple[float, ...] = (1.0,)
    selection_verbose: bool = True


def load_prepared_alignment_artifacts(
    cache_path: str | Path,
    *,
    expected_spec: dict[str, object] | None = None,
) -> dict[str, object] | None:
    """Load cached IOI signature artifacts when the on-disk spec matches."""
    path = Path(cache_path)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    cached_spec = payload.get("cache_spec")
    if expected_spec is not None and cached_spec != expected_spec:
        return None
    base_logits = payload.get("base_logits")
    site_signatures = payload.get("site_signatures")
    if not isinstance(base_logits, torch.Tensor) or not isinstance(site_signatures, torch.Tensor):
        return None
    return {
        "base_logits": base_logits.detach().cpu(),
        "site_signatures": site_signatures.detach().cpu(),
        "prepare_runtime_seconds": float(payload.get("prepare_runtime_seconds", 0.0)),
        "cache_spec": cached_spec,
        "cache_path": str(path),
        "loaded_from_disk": True,
    }


def save_prepared_alignment_artifacts(
    cache_path: str | Path,
    *,
    prepared_artifacts: dict[str, object],
    cache_spec: dict[str, object],
) -> None:
    """Persist reusable IOI signature artifacts for future epsilon sweeps and reruns."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "cache_spec": cache_spec,
            "prepare_runtime_seconds": float(prepared_artifacts.get("prepare_runtime_seconds", 0.0)),
            "base_logits": prepared_artifacts["base_logits"].detach().cpu(),
            "site_signatures": prepared_artifacts["site_signatures"].detach().cpu(),
            "saved_with": "ioi_signature_cache_v1",
            "saved_spec_json": json.dumps(cache_spec, sort_keys=True),
        },
        path,
    )


def prepare_alignment_artifacts(
    *,
    model,
    fit_bank: IOIPairBank,
    sites: list[ResidualSite],
    device: torch.device | str,
    config: OTConfig,
) -> dict[str, torch.Tensor]:
    """Build reusable factual logits and neural site signatures for one OT/UOT run."""
    device = torch.device(device)
    start = perf_counter()
    base_logits = collect_base_logits(
        model=model,
        bank=fit_bank,
        batch_size=config.batch_size,
        device=device,
    )
    site_signatures = collect_site_signatures(
        model=model,
        bank=fit_bank,
        sites=sites,
        base_logits=base_logits,
        batch_size=config.batch_size,
        device=device,
        signature_mode=config.signature_mode,
        show_progress=config.selection_verbose,
    )
    return {
        "base_logits": base_logits,
        "site_signatures": site_signatures,
        "prepare_runtime_seconds": float(perf_counter() - start),
        "loaded_from_disk": False,
    }


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


def solve_ot_transport(
    variable_signature: torch.Tensor, site_signatures: torch.Tensor, config: OTConfig
) -> tuple[np.ndarray, dict[str, object]]:
    variable_signature = variable_signature.reshape(1, -1)
    site_signatures = site_signatures.reshape(site_signatures.shape[0], -1)
    p = np.ones(variable_signature.shape[0], dtype=np.float64) / float(variable_signature.shape[0])
    q = np.ones(site_signatures.shape[0], dtype=np.float64) / float(site_signatures.shape[0])
    transport_tensor, transport_cost = sinkhorn_uniform_ot(
        variable_signature,
        site_signatures,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        temperature=1.0,
        tol=float(config.tol),
    )
    transport = transport_tensor.detach().cpu().numpy()
    meta = {
        "method": "ot",
        "regularization_used": float(config.epsilon),
        "epsilon_config": float(config.epsilon),
        "transport_cost": float(transport_cost),
        **_transport_validation_stats(transport, p, q),
    }
    if _is_valid_balanced_transport(transport, p, q, float(config.tol)):
        return transport, meta
    meta.update({"failed": True, "failure_reason": "invalid_balanced_transport"})
    return transport, meta


def solve_uot_transport(
    variable_signature: torch.Tensor, site_signatures: torch.Tensor, config: OTConfig
) -> tuple[np.ndarray, dict[str, object]]:
    variable_signature = variable_signature.reshape(1, -1)
    site_signatures = site_signatures.reshape(site_signatures.shape[0], -1)
    transport_tensor, info = sinkhorn_unbalanced_ot(
        variable_signature,
        site_signatures,
        epsilon=float(config.epsilon),
        n_iter=int(config.max_iter),
        temperature=1.0,
        tau_abstract=float(config.uot_beta_abstract),
        tau_neural=float(config.uot_beta_neural),
    )
    transport = transport_tensor.detach().cpu().numpy()
    meta = {
        "method": "uot",
        "regularization_used": float(config.epsilon),
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
    bank: IOIPairBank,
    sites: list[ResidualSite],
    selected_transport: np.ndarray,
    top_k: int,
    strength: float,
    batch_size: int,
    device: torch.device,
    tokenizer,
    include_details: bool = False,
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
        **_metrics_from_logits(logits, bank, tokenizer=tokenizer),
    }
    if include_details:
        record["prediction_details"] = _prediction_details_from_logits(logits, bank, tokenizer=tokenizer)
    return record, ranking


def _select_hyperparameters(
    *,
    model,
    calibration_bank: IOIPairBank,
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
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] calibration start variable={calibration_bank.target_var} "
            f"signature_mode={config.signature_mode} top_k_values={list(top_k_values)} "
            f"lambda_values={list(config.lambda_values)}"
        )
    calibration_candidates = [(int(top_k), float(strength)) for top_k in top_k_values for strength in config.lambda_values]
    candidate_iterator = calibration_candidates
    if config.selection_verbose and tqdm is not None:
        candidate_iterator = tqdm(
            calibration_candidates,
            desc=f"{config.method.upper()} calibration sweep ({calibration_bank.target_var})",
            leave=False,
        )
    for top_k, strength in candidate_iterator:
        truncated = truncate_transport_rows(normalized_transport, top_k, renormalize=True)
        result, ranking = _evaluate_soft_intervention(
            model=model,
            bank=calibration_bank,
            sites=sites,
            selected_transport=truncated,
            top_k=top_k,
            strength=strength,
            batch_size=batch_size,
            device=device,
            tokenizer=tokenizer,
            include_details=False,
        )
        candidate = {
            "top_k": top_k,
            "lambda": strength,
            "result": result,
            "ranking": ranking,
            "exact_acc": float(result["exact_acc"]),
        }
        sweep_records.append(candidate)
        if best is None or float(candidate["exact_acc"]) > float(best["exact_acc"]):
            best = candidate
            if config.selection_verbose:
                print(
                    f"[{config.method.upper()}] new best variable={calibration_bank.target_var} "
                    f"top_k={int(top_k)} lambda={float(strength):g} "
                    f"calibration_exact_acc={float(candidate['exact_acc']):.4f}"
                )
    if best is None:
        raise RuntimeError(f"Failed to select OT/UOT hyperparameters for {calibration_bank.target_var}")
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] selected variable={calibration_bank.target_var} "
            f"top_k={int(best['top_k'])} lambda={float(best['lambda']):g} "
            f"calibration_exact_acc={float(best['exact_acc']):.4f}"
        )
    return best, sweep_records


def run_alignment_pipeline(
    *,
    model,
    fit_bank: IOIPairBank,
    calibration_bank: IOIPairBank,
    holdout_bank: IOIPairBank,
    sites: list[ResidualSite],
    device: torch.device | str,
    tokenizer,
    config: OTConfig,
    prepared_artifacts: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    """Run OT or UOT for one IOI target variable."""
    device = torch.device(device)
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] start variable={fit_bank.target_var} "
            f"signature_mode={config.signature_mode} candidate_sites={len(sites)} "
            f"epsilon={float(config.epsilon):g}"
        )
    if prepared_artifacts is None:
        prepared_artifacts = prepare_alignment_artifacts(
            model=model,
            fit_bank=fit_bank,
            sites=sites,
            device=device,
            config=config,
        )
    base_logits = prepared_artifacts["base_logits"]
    site_signatures = prepared_artifacts["site_signatures"]

    variable_signature = build_variable_signature(fit_bank, config.signature_mode)
    if config.method == "ot":
        transport, transport_meta = solve_ot_transport(variable_signature, site_signatures, config)
    elif config.method == "uot":
        transport, transport_meta = solve_uot_transport(variable_signature, site_signatures, config)
    else:
        raise ValueError(f"Unsupported IOI transport method {config.method!r}")
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
    selected_calibration_result, selected_calibration_ranking = _evaluate_soft_intervention(
        model=model,
        bank=calibration_bank,
        sites=sites,
        selected_transport=selected_transport,
        top_k=top_k,
        strength=strength,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        include_details=True,
    )
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
        include_details=True,
    )
    holdout_result["method"] = config.method
    holdout_result["selection_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["calibration_exact_acc"] = float(selected["result"]["exact_acc"])
    holdout_result["signature_mode"] = str(config.signature_mode)
    holdout_result["selected_transport_nonzero"] = int((selected_transport[0] > 0.0).sum())
    if config.selection_verbose:
        print(
            f"[{config.method.upper()}] holdout variable={fit_bank.target_var} "
            f"top_k={top_k} lambda={strength:g} exact_acc={float(holdout_result['exact_acc']):.4f}"
        )
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
        "selected_calibration_result": selected_calibration_result,
        "selected_calibration_ranking": selected_calibration_ranking,
        "ranking": holdout_ranking,
        "calibration_sweep": calibration_sweep,
        "results": [holdout_result],
    }
