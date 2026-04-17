"""DAS search over MCQA Gemma residual-stream sites."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from .data import MCQAPairBank, MCQAPairDataset
from .intervention import DASSubspaceIntervention, run_das_residual_intervention
from .metrics import (
    cross_entropy_for_das,
    das_metrics_from_logits,
    das_prediction_details_from_logits,
)
from .sites import ResidualSite


@dataclass(frozen=True)
class DASConfig:
    """Hyperparameters controlling MCQA DAS search."""

    batch_size: int = 16
    max_epochs: int = 5
    min_epochs: int = 1
    plateau_patience: int = 2
    plateau_rel_delta: float = 5e-3
    learning_rate: float = 1e-3
    subspace_dims: tuple[int, ...] | None = None
    store_candidate_holdout_metrics: bool = True
    verbose: bool = True


def train_das_candidate(
    *,
    model,
    bank: MCQAPairBank,
    site: ResidualSite,
    subspace_dim: int,
    batch_size: int,
    max_epochs: int,
    min_epochs: int,
    plateau_patience: int,
    plateau_rel_delta: float,
    learning_rate: float,
    device: torch.device,
    verbose: bool = True,
) -> tuple[DASSubspaceIntervention, list[float]]:
    intervention = DASSubspaceIntervention(hidden_size=int(model.config.hidden_size), subspace_dim=int(subspace_dim)).to(device)
    optimizer = torch.optim.Adam(intervention.parameters(), lr=float(learning_rate))
    loader = DataLoader(MCQAPairDataset(bank), batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    best_loss = None
    plateau_steps = 0
    for epoch_index in range(int(max_epochs)):
        epoch_losses = []
        for batch in loader:
            logits = run_das_residual_intervention(
                model=model,
                base_input_ids=batch["base_input_ids"].to(device),
                base_attention_mask=batch["base_attention_mask"].to(device),
                source_input_ids=batch["source_input_ids"].to(device),
                source_attention_mask=batch["source_attention_mask"].to(device),
                site=site,
                intervention=intervention,
                base_position_by_id={
                    key: value.to(device) for key, value in batch["base_positions"].items()
                },
                source_position_by_id={
                    key: value.to(device) for key, value in batch["source_positions"].items()
                },
            )
            mini_bank = MCQAPairBank(
                split=bank.split,
                target_var=bank.target_var,
                dataset_names=bank.dataset_names,
                base_input_ids=batch["base_input_ids"],
                base_attention_mask=batch["base_attention_mask"],
                source_input_ids=batch["source_input_ids"],
                source_attention_mask=batch["source_attention_mask"],
                labels=batch["labels"],
                base_inputs=[],
                source_inputs=[],
                base_outputs=[],
                source_outputs=[],
                base_position_by_id={key: value.detach().cpu() for key, value in batch["base_positions"].items()},
                source_position_by_id={key: value.detach().cpu() for key, value in batch["source_positions"].items()},
                symbol_token_ids=batch["symbol_token_ids"],
                symbol_variant_token_ids=batch["symbol_variant_token_ids"],
                source_symbol_token_ids=batch["source_symbol_token_ids"],
                source_symbol_variant_token_ids=batch["source_symbol_variant_token_ids"],
                alphabet_token_ids=batch["alphabet_token_ids"],
                alphabet_variant_token_ids=batch["alphabet_variant_token_ids"],
                canonical_answer_token_ids=bank.canonical_answer_token_ids,
                answer_token_ids=batch["answer_token_id"].view(-1),
                base_answer_token_ids=batch["base_answer_token_id"].view(-1),
                changed_mask=torch.ones_like(batch["labels"], dtype=torch.bool),
                expected_answer_texts=batch["expected_answer_text"],
            )
            loss = cross_entropy_for_das(logits, mini_bank)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(epoch_loss)
        if verbose:
            print(
                f"[DAS] epoch {epoch_index + 1}/{int(max_epochs)} "
                f"variable={bank.target_var} site={site.label} dim={int(subspace_dim)} "
                f"loss={float(epoch_loss):.6f}"
            )
        if best_loss is None or epoch_loss < float(best_loss) * (1.0 - float(plateau_rel_delta)):
            best_loss = epoch_loss
            plateau_steps = 0
        else:
            plateau_steps += 1
        if epoch_index + 1 >= int(min_epochs) and plateau_steps >= int(plateau_patience):
            break
    return intervention, losses


def evaluate_das_candidate(
    *,
    model,
    bank: MCQAPairBank,
    site: ResidualSite,
    intervention: DASSubspaceIntervention,
    batch_size: int,
    device: torch.device,
    tokenizer,
    return_details: bool = False,
) -> dict[str, object]:
    loader = DataLoader(MCQAPairDataset(bank), batch_size=batch_size, shuffle=False)
    logits_all = []
    for batch in loader:
        logits = run_das_residual_intervention(
            model=model,
            base_input_ids=batch["base_input_ids"].to(device),
            base_attention_mask=batch["base_attention_mask"].to(device),
            source_input_ids=batch["source_input_ids"].to(device),
            source_attention_mask=batch["source_attention_mask"].to(device),
            site=site,
            intervention=intervention,
            base_position_by_id={key: value.to(device) for key, value in batch["base_positions"].items()},
            source_position_by_id={key: value.to(device) for key, value in batch["source_positions"].items()},
        )
        logits_all.append(logits.detach().cpu())
    full_logits = torch.cat(logits_all, dim=0)
    metrics = das_metrics_from_logits(full_logits, bank, tokenizer=tokenizer)
    if return_details:
        metrics["prediction_details"] = das_prediction_details_from_logits(full_logits, bank, tokenizer=tokenizer)
    return metrics


def run_das_pipeline(
    *,
    model,
    train_bank: MCQAPairBank,
    calibration_bank: MCQAPairBank,
    holdout_bank: MCQAPairBank,
    sites: list[ResidualSite],
    device: torch.device | str,
    tokenizer,
    config: DASConfig,
) -> dict[str, object]:
    device = torch.device(device)
    hidden_size = int(model.config.hidden_size)
    subspace_dims = tuple(range(1, hidden_size + 1)) if config.subspace_dims is None else tuple(config.subspace_dims)
    search_records = []
    best = None
    best_intervention = None
    best_site = None
    total_candidates = len(sites) * len(subspace_dims)
    candidate_index = 0
    if config.verbose:
        print(
            f"[DAS] start variable={train_bank.target_var} "
            f"sites={len(sites)} subspace_dims={list(subspace_dims)} total_candidates={total_candidates}"
        )
    for site in sites:
        for subspace_dim in subspace_dims:
            candidate_index += 1
            if config.verbose:
                print(
                    f"[DAS] candidate {candidate_index}/{total_candidates} "
                    f"variable={train_bank.target_var} site={site.label} dim={int(subspace_dim)}"
                )
            intervention, loss_history = train_das_candidate(
                model=model,
                bank=train_bank,
                site=site,
                subspace_dim=int(subspace_dim),
                batch_size=config.batch_size,
                max_epochs=config.max_epochs,
                min_epochs=config.min_epochs,
                plateau_patience=config.plateau_patience,
                plateau_rel_delta=config.plateau_rel_delta,
                learning_rate=config.learning_rate,
                device=device,
                verbose=config.verbose,
            )
            calibration_metrics = evaluate_das_candidate(
                model=model,
                bank=calibration_bank,
                site=site,
                intervention=intervention,
                batch_size=config.batch_size,
                device=device,
                tokenizer=tokenizer,
                return_details=False,
            )
            holdout_metrics_for_candidate = None
            if config.store_candidate_holdout_metrics:
                holdout_metrics_for_candidate = evaluate_das_candidate(
                    model=model,
                    bank=holdout_bank,
                    site=site,
                    intervention=intervention,
                    batch_size=config.batch_size,
                    device=device,
                    tokenizer=tokenizer,
                    return_details=False,
                )
            record = {
                "method": "das",
                "variable": train_bank.target_var,
                "split": calibration_bank.split,
                "site_label": site.label,
                "layer": int(site.layer),
                "token_position_id": site.token_position_id,
                "subspace_dim": int(subspace_dim),
                "selection_exact_acc": float(calibration_metrics["exact_acc"]),
                "calibration_exact_acc": float(calibration_metrics["exact_acc"]),
                "train_epochs_ran": len(loss_history),
                "train_loss_history": loss_history,
            }
            if holdout_metrics_for_candidate is not None:
                record["holdout_exact_acc"] = float(holdout_metrics_for_candidate["exact_acc"])
                if "decoded_answer_acc" in holdout_metrics_for_candidate:
                    record["holdout_decoded_answer_acc"] = float(holdout_metrics_for_candidate["decoded_answer_acc"])
            search_records.append(record)
            if config.verbose:
                message = (
                    f"[DAS] calibration variable={train_bank.target_var} site={site.label} "
                    f"dim={int(subspace_dim)} epochs={len(loss_history)} "
                    f"exact_acc={float(calibration_metrics['exact_acc']):.4f}"
                )
                if holdout_metrics_for_candidate is not None:
                    message += f" holdout_exact_acc={float(holdout_metrics_for_candidate['exact_acc']):.4f}"
                print(message)
            if best is None or float(record["selection_exact_acc"]) > float(best["selection_exact_acc"]):
                best = record
                best_intervention = intervention
                best_site = site
                if config.verbose:
                    print(
                        f"[DAS] new best variable={train_bank.target_var} "
                        f"site={site.label} dim={int(subspace_dim)} "
                        f"calibration_exact_acc={float(record['selection_exact_acc']):.4f}"
                    )
    if best is None or best_intervention is None or best_site is None:
        raise RuntimeError(f"Failed to select a DAS candidate for {train_bank.target_var}")
    holdout_metrics = evaluate_das_candidate(
        model=model,
        bank=holdout_bank,
        site=best_site,
        intervention=best_intervention,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        return_details=True,
    )
    selected_calibration_metrics = evaluate_das_candidate(
        model=model,
        bank=calibration_bank,
        site=best_site,
        intervention=best_intervention,
        batch_size=config.batch_size,
        device=device,
        tokenizer=tokenizer,
        return_details=True,
    )
    result = {
        **best,
        "split": holdout_bank.split,
        **holdout_metrics,
    }
    if config.verbose:
        print(
            f"[DAS] holdout variable={train_bank.target_var} "
            f"site={best_site.label} dim={int(best['subspace_dim'])} "
            f"exact_acc={float(holdout_metrics['exact_acc']):.4f}"
        )
    return {
        "target_var": train_bank.target_var,
        "training_stopping_rule": {
            "max_epochs": config.max_epochs,
            "min_epochs": config.min_epochs,
            "plateau_patience": config.plateau_patience,
            "plateau_rel_delta": config.plateau_rel_delta,
        },
        "selected_calibration_metrics": selected_calibration_metrics,
        "search_records": {train_bank.target_var: search_records},
        "results": [result],
    }
