"""IOI residual-site effect signatures."""

from __future__ import annotations

import torch
import torch.nn.functional as F
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from .data import IOIPairBank
from .intervention import forward_factual_logits, run_soft_residual_intervention
from .sites import ResidualSite


def _per_example_kl(counterfactual_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    base_log_probs = torch.log_softmax(base_logits, dim=-1)
    counterfactual_log_probs = torch.log_softmax(counterfactual_logits, dim=-1)
    counterfactual_probs = counterfactual_log_probs.exp()
    return torch.sum(counterfactual_probs * (counterfactual_log_probs - base_log_probs), dim=-1)


def _gather_choice_logits(logits: torch.Tensor, bank: IOIPairBank) -> torch.Tensor:
    """Gather the two source-choice token logits: shape [N, 2]."""
    return torch.gather(logits, 1, bank.source_choice_token_ids.to(logits.device))


def signature_from_logits(
    *,
    counterfactual_logits: torch.Tensor,
    base_logits: torch.Tensor,
    bank: IOIPairBank,
    signature_mode: str,
) -> torch.Tensor:
    """Convert factual/counterfactual logits into one site-effect signature."""
    if signature_mode == "whole_vocab_kl_t1":
        return _per_example_kl(counterfactual_logits, base_logits).reshape(-1)
    if signature_mode == "answer_logit_delta":
        counterfactual_choice = _gather_choice_logits(counterfactual_logits, bank)
        base_choice = _gather_choice_logits(base_logits, bank)
        return (counterfactual_choice - base_choice).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode!r}")


def build_variable_signature(bank: IOIPairBank, signature_mode: str) -> torch.Tensor:
    """Build the abstract-variable signature for one IOI target variable."""
    if signature_mode == "whole_vocab_kl_t1":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_logit_delta":
        source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=2).to(torch.float32)
        base_answer_indices = torch.tensor(
            [int(output["answer_index"]) for output in bank.base_outputs],
            dtype=torch.long,
        )
        base_onehot = F.one_hot(base_answer_indices, num_classes=2).to(torch.float32)
        return (source_onehot - base_onehot).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode!r}")


def collect_base_logits(
    *,
    model,
    bank: IOIPairBank,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Collect factual last-token logits for a bank."""
    outputs = []
    with torch.no_grad():
        for start in range(0, bank.size, batch_size):
            end = min(start + batch_size, bank.size)
            logits = forward_factual_logits(
                model=model,
                input_ids=bank.base_input_ids[start:end].to(device),
                attention_mask=bank.base_attention_mask[start:end].to(device),
            )
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def collect_site_signatures(
    *,
    model,
    bank: IOIPairBank,
    sites: list[ResidualSite],
    base_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
    signature_mode: str,
    show_progress: bool = False,
) -> torch.Tensor:
    """Measure each residual site's intervention effect signature."""
    signatures = []
    site_iterator = sites
    if show_progress and tqdm is not None:
        site_iterator = tqdm(sites, desc=f"Building {signature_mode} signatures", leave=False)
    with torch.no_grad():
        for site in site_iterator:
            site_logits_chunks = []
            for start in range(0, bank.size, batch_size):
                end = min(start + batch_size, bank.size)
                logits = run_soft_residual_intervention(
                    model=model,
                    base_input_ids=bank.base_input_ids[start:end].to(device),
                    base_attention_mask=bank.base_attention_mask[start:end].to(device),
                    source_input_ids=bank.source_input_ids[start:end].to(device),
                    source_attention_mask=bank.source_attention_mask[start:end].to(device),
                    site_weights={site: 1.0},
                    strength=1.0,
                    base_position_by_id={
                        key: value[start:end] for key, value in bank.base_position_by_id.items()
                    },
                    source_position_by_id={
                        key: value[start:end] for key, value in bank.source_position_by_id.items()
                    },
                )
                site_logits_chunks.append(logits.detach().cpu())
            site_logits = torch.cat(site_logits_chunks, dim=0)
            signatures.append(
                signature_from_logits(
                    counterfactual_logits=site_logits,
                    base_logits=base_logits,
                    bank=bank,
                    signature_mode=signature_mode,
                )
            )
    return torch.stack(signatures, dim=0)
