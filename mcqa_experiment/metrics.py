"""MCQA-specific label extraction and reporting metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import MCQAPairBank


def gather_variable_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Project full-vocab logits into the task logits for the chosen target variable."""
    if bank.target_var == "answer_pointer":
        gathered = torch.gather(logits, dim=1, index=bank.symbol_token_ids.to(logits.device))
        return gathered
    if bank.target_var == "answer":
        return torch.gather(logits, dim=1, index=bank.source_symbol_token_ids.to(logits.device))
    raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")


def cross_entropy_for_bank(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute supervised cross-entropy for the bank's target variable."""
    target_logits = gather_variable_logits(logits, bank)
    return F.cross_entropy(target_logits, bank.labels.to(target_logits.device))


def metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, float]:
    """Compute exact accuracy and optional decoded answer accuracy."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    exact_acc = float((predictions == bank.labels.to(predictions.device)).float().mean().item())
    metrics = {"exact_acc": exact_acc}
    if bank.target_var == "answer" and tokenizer is not None:
        token_predictions = torch.gather(
            bank.source_symbol_token_ids.to(logits.device),
            dim=1,
            index=predictions.view(-1, 1),
        ).view(-1)
        decoded_predictions = [tokenizer.decode([int(token_id)]) for token_id in token_predictions.detach().cpu().tolist()]
        decoded_acc = 0.0
        if decoded_predictions:
            decoded_acc = sum(
                int(expected in decoded)
                for expected, decoded in zip(bank.expected_answer_texts, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
    return metrics


def build_variable_signature(bank: MCQAPairBank, signature_mode: str) -> torch.Tensor:
    """Build the abstract-variable signature for one MCQA target variable."""
    if signature_mode == "whole_vocab_kl_t1":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_logit_delta":
        if bank.target_var == "answer_pointer":
            base_pointer = torch.tensor([int(output["answer_pointer"]) for output in bank.base_outputs], dtype=torch.long)
            base_onehot = F.one_hot(base_pointer, num_classes=4).to(torch.float32)
            source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=4).to(torch.float32)
            return (source_onehot - base_onehot).reshape(-1)
        if bank.target_var == "answer":
            source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=4).to(torch.float32)
            base_matches = bank.source_symbol_token_ids.eq(bank.base_answer_token_ids.view(-1, 1))
            base_onehot = base_matches.to(torch.float32)
            return (source_onehot - base_onehot).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
