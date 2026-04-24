"""Residual-stream intervention helpers for IOI runs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .sites import ResidualSite


def resolve_transformer_layers(model) -> list[nn.Module]:
    """Resolve the ordered transformer blocks for common causal LM wrappers."""
    for path in (
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ):
        current = model
        found = True
        for attribute in path:
            if not hasattr(current, attribute):
                found = False
                break
            current = getattr(current, attribute)
        if found:
            return list(current)
    raise ValueError(f"Could not locate transformer layer stack on model type {type(model)!r}")


def get_num_layers(model) -> int:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    return len(resolve_transformer_layers(model))


def get_hidden_size(model) -> int:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    raise ValueError(f"Could not resolve hidden_size from model config {type(model.config)!r}")


def gather_last_token_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Gather next-token logits at the final non-pad token for each example."""
    last_indices = attention_mask.sum(dim=1).to(torch.long) - 1
    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, last_indices]


def forward_factual_logits(
    *,
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run the factual model and return last-token logits."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    return gather_last_token_logits(outputs.logits, attention_mask)


class DASSubspaceIntervention(nn.Module):
    """Low-rank rotated-space swap on one residual vector."""

    def __init__(self, hidden_size: int, subspace_dim: int) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.subspace_dim = int(subspace_dim)
        linear = nn.Linear(self.hidden_size, self.subspace_dim, bias=False)
        self.project = torch.nn.utils.parametrizations.orthogonal(linear)

    def forward(self, base_vectors: torch.Tensor, source_vectors: torch.Tensor) -> torch.Tensor:
        basis = self.project.weight
        compute_dtype = basis.dtype
        base_vectors_compute = base_vectors.to(compute_dtype)
        source_vectors_compute = source_vectors.to(compute_dtype)
        base_features = base_vectors_compute @ basis.t()
        source_features = source_vectors_compute @ basis.t()
        updated = base_vectors_compute + (source_features - base_features) @ basis
        return updated.to(base_vectors.dtype)


def _collect_source_hidden_states(
    *,
    model,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    target_layers: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    outputs = model(
        input_ids=source_input_ids,
        attention_mask=source_attention_mask,
        use_cache=False,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states
    return {int(layer): hidden_states[int(layer) + 1] for layer in target_layers}


def run_das_residual_intervention(
    *,
    model,
    base_input_ids: torch.Tensor,
    base_attention_mask: torch.Tensor,
    source_input_ids: torch.Tensor,
    source_attention_mask: torch.Tensor,
    site: ResidualSite,
    intervention: DASSubspaceIntervention,
    base_position_by_id: dict[str, torch.Tensor],
    source_position_by_id: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply one trainable DAS-style rotated-space swap at a residual-stream site."""
    layers = resolve_transformer_layers(model)
    source_hidden_by_layer = _collect_source_hidden_states(
        model=model,
        source_input_ids=source_input_ids,
        source_attention_mask=source_attention_mask,
        target_layers=(site.layer,),
    )

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        batch_size = hidden.shape[0]
        batch_indices = torch.arange(batch_size, device=hidden.device)
        source_hidden = source_hidden_by_layer[int(site.layer)].to(hidden.device)
        base_positions = base_position_by_id[site.token_position_id].to(hidden.device)
        source_positions = source_position_by_id[site.token_position_id].to(hidden.device)
        base_vectors = hidden[batch_indices, base_positions]
        source_vectors = source_hidden[batch_indices, source_positions]
        hidden[batch_indices, base_positions] = intervention(base_vectors, source_vectors)
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    handle = layers[int(site.layer)].register_forward_hook(hook)
    try:
        outputs = model(
            input_ids=base_input_ids,
            attention_mask=base_attention_mask,
            use_cache=False,
        )
    finally:
        handle.remove()
    return gather_last_token_logits(outputs.logits, base_attention_mask)
