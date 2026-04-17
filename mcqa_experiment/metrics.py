"""MCQA-specific label extraction and reporting metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import MCQAPairBank, ALPHABET_LABELS


def _gather_variant_logits(logits: torch.Tensor, variant_token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, num_classes, num_variants = variant_token_ids.shape
    gathered = torch.gather(
        logits,
        dim=1,
        index=variant_token_ids.to(logits.device).reshape(batch_size, num_classes * num_variants),
    )
    gathered = gathered.reshape(batch_size, num_classes, num_variants)
    return gathered.max(dim=-1).values


def gather_variable_logits(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Project full-vocab logits into the task logits for the chosen target variable."""
    if bank.target_var in {"answer_pointer", "answer"}:
        return _gather_variant_logits(logits, bank.alphabet_variant_token_ids)
    raise ValueError(f"Unsupported MCQA target variable {bank.target_var}")


def cross_entropy_for_bank(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute supervised cross-entropy for the bank's target variable."""
    target_logits = gather_variable_logits(logits, bank)
    return F.cross_entropy(target_logits, bank.labels.to(target_logits.device))


def cross_entropy_for_das(logits: torch.Tensor, bank: MCQAPairBank) -> torch.Tensor:
    """Compute DAS training loss on full next-token logits."""
    return F.cross_entropy(logits, bank.answer_token_ids.to(logits.device))


def metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, float]:
    """Compute exact accuracy and optional decoded answer accuracy."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    exact_acc = float((predictions == bank.labels.to(predictions.device)).float().mean().item())
    metrics = {"exact_acc": exact_acc}
    if tokenizer is not None:
        token_predictions = torch.gather(
            bank.alphabet_token_ids.to(logits.device),
            dim=1,
            index=predictions.view(-1, 1),
        ).view(-1)
        decoded_predictions = [tokenizer.decode([int(token_id)]) for token_id in token_predictions.detach().cpu().tolist()]
        decoded_acc = 0.0
        if decoded_predictions:
            decoded_acc = sum(
                int(str(expected).strip() == str(decoded).strip())
                for expected, decoded in zip(bank.expected_answer_texts, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
    return metrics


def das_metrics_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, float]:
    """Compute DAS metrics directly on full-vocab next-token predictions."""
    predictions = logits.argmax(dim=-1)
    labels = bank.answer_token_ids.to(predictions.device)
    exact_acc = float((predictions == labels).float().mean().item())
    metrics = {"exact_acc": exact_acc}
    if tokenizer is not None:
        decoded_predictions = [tokenizer.decode([int(token_id)]) for token_id in predictions.detach().cpu().tolist()]
        decoded_acc = 0.0
        if decoded_predictions:
            decoded_acc = sum(
                int(str(expected).strip() == str(decoded).strip())
                for expected, decoded in zip(bank.expected_answer_texts, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
    return metrics


def prediction_details_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Return parse-friendly per-example prediction details for one bank."""
    target_logits = gather_variable_logits(logits, bank)
    predictions = target_logits.argmax(dim=-1)
    labels = bank.labels.to(predictions.device)
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "target_logits": target_logits.detach().cpu().tolist(),
        "base_raw_inputs": [str(item["raw_input"]) for item in bank.base_inputs],
        "source_raw_inputs": [str(item["raw_input"]) for item in bank.source_inputs],
        "expected_answer_texts": list(bank.expected_answer_texts),
    }
    predicted_token_ids = torch.gather(
        bank.alphabet_token_ids.to(logits.device),
        dim=1,
        index=predictions.view(-1, 1),
    ).view(-1)
    details["predicted_token_ids"] = predicted_token_ids.detach().cpu().tolist()
    details["target_token_ids"] = bank.answer_token_ids.detach().cpu().tolist()
    if tokenizer is not None:
        details["predicted_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predicted_token_ids.detach().cpu().tolist()
        ]
    return details


def das_prediction_details_from_logits(logits: torch.Tensor, bank: MCQAPairBank, tokenizer=None) -> dict[str, object]:
    """Return parse-friendly DAS prediction details from full-vocab logits."""
    predictions = logits.argmax(dim=-1)
    labels = bank.answer_token_ids.to(predictions.device)
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "base_raw_inputs": [str(item["raw_input"]) for item in bank.base_inputs],
        "source_raw_inputs": [str(item["raw_input"]) for item in bank.source_inputs],
        "expected_answer_texts": list(bank.expected_answer_texts),
        "target_token_ids": bank.answer_token_ids.detach().cpu().tolist(),
    }
    if tokenizer is not None:
        details["predicted_text"] = [
            tokenizer.decode([int(token_id)]) for token_id in predictions.detach().cpu().tolist()
        ]
    return details


def build_variable_signature(bank: MCQAPairBank, signature_mode: str) -> torch.Tensor:
    """Build the abstract-variable signature for one MCQA target variable."""
    if signature_mode == "whole_vocab_kl_t1":
        return bank.changed_mask.to(torch.float32)
    if signature_mode == "answer_logit_delta":
        source_onehot = F.one_hot(bank.labels.to(torch.long), num_classes=26).to(torch.float32)
        base_answer_indices = torch.tensor(
            [ALPHABET_LABELS.index(str(output["answer"]).strip()) for output in bank.base_outputs],
            dtype=torch.long,
        )
        base_onehot = F.one_hot(base_answer_indices, num_classes=26).to(torch.float32)
        return (source_onehot - base_onehot).reshape(-1)
    raise ValueError(f"Unsupported signature_mode={signature_mode}")
