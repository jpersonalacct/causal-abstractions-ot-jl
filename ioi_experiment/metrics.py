"""IOI-specific DAS metrics."""

from __future__ import annotations

import torch

from .data import IOIPairBank


def das_metrics_from_logits(
    logits: torch.Tensor, bank: IOIPairBank, tokenizer=None
) -> dict[str, object]:
    """Compute DAS metrics: token exact match and choice-constrained accuracy.

    exact_acc: argmax of full-vocab logits matches the source answer token.
    choice_acc: among the two source choice tokens, the higher-logit one matches labels.
    """
    predictions = logits.argmax(dim=-1)
    labels_token = bank.answer_token_ids.to(predictions.device)
    exact_acc = float((predictions == labels_token).float().mean().item())

    source_choice_ids = bank.source_choice_token_ids.to(logits.device)  # [N, 2]
    choice_logits = torch.gather(logits, 1, source_choice_ids)  # [N, 2]
    choice_predictions = choice_logits.argmax(dim=-1)  # [N]
    choice_acc = float(
        (choice_predictions == bank.labels.to(choice_predictions.device)).float().mean().item()
    )

    metrics: dict[str, object] = {"exact_acc": exact_acc, "choice_acc": choice_acc}
    if tokenizer is not None:
        decoded_predictions = [
            tokenizer.decode([int(token_id)]) for token_id in predictions.detach().cpu().tolist()
        ]
        decoded_acc = 0.0
        if decoded_predictions:
            decoded_acc = sum(
                int(str(expected).strip() == str(decoded).strip())
                for expected, decoded in zip(bank.expected_answer_texts, decoded_predictions)
            ) / len(decoded_predictions)
        metrics["decoded_answer_acc"] = float(decoded_acc)
    return metrics


def das_prediction_details_from_logits(
    logits: torch.Tensor, bank: IOIPairBank, tokenizer=None
) -> dict[str, object]:
    """Return per-example prediction details for IOI DAS evaluation."""
    predictions = logits.argmax(dim=-1)
    labels = bank.answer_token_ids.to(predictions.device)
    source_choice_ids = bank.source_choice_token_ids.to(logits.device)  # [N, 2]
    choice_logits = torch.gather(logits, 1, source_choice_ids)  # [N, 2]
    choice_predictions = choice_logits.argmax(dim=-1)  # [N]
    details: dict[str, object] = {
        "labels": labels.detach().cpu().tolist(),
        "predictions": predictions.detach().cpu().tolist(),
        "correct": (predictions == labels).detach().cpu().to(torch.int64).tolist(),
        "choice_labels": bank.labels.detach().cpu().tolist(),
        "choice_predictions": choice_predictions.detach().cpu().tolist(),
        "choice_correct": (
            choice_predictions == bank.labels.to(choice_predictions.device)
        ).detach().cpu().to(torch.int64).tolist(),
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
