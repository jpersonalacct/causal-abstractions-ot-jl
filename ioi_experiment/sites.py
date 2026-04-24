"""Residual-stream site enumeration for IOI transformer interventions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidualSite:
    """One residual-stream block identified by layer, token-position, and dim range."""

    layer: int
    token_position_id: str
    dim_start: int
    dim_end: int

    @property
    def label(self) -> str:
        return f"L{int(self.layer)}:{self.token_position_id}[{int(self.dim_start)}:{int(self.dim_end)}]"


def enumerate_residual_sites(
    *,
    num_layers: int,
    hidden_size: int,
    token_position_ids: tuple[str, ...],
    resolution: int | None = 1,
    layers: tuple[int, ...] | None = None,
    selected_token_position_ids: tuple[str, ...] | None = None,
) -> list[ResidualSite]:
    """Enumerate residual-stream dimension blocks for the IOI sweep."""
    layer_ids = tuple(range(int(num_layers))) if layers is None else tuple(int(layer) for layer in layers)
    position_ids = token_position_ids if selected_token_position_ids is None else selected_token_position_ids
    block_width = int(hidden_size) if resolution is None else max(1, int(resolution))
    dim_blocks = [
        (dim_start, min(dim_start + block_width, int(hidden_size)))
        for dim_start in range(0, int(hidden_size), block_width)
    ]
    return [
        ResidualSite(
            layer=layer,
            token_position_id=position_id,
            dim_start=dim_start,
            dim_end=dim_end,
        )
        for layer in layer_ids
        for position_id in position_ids
        for dim_start, dim_end in dim_blocks
    ]
