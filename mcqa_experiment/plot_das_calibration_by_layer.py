"""Render a DAS calibration heatmap (layer x subspace_dim) from a saved MCQA DAS JSON payload."""

from __future__ import annotations

import json
from pathlib import Path


INPUT_PATH = Path("results/4-9 mcqa gemma-2/20260409_144935_mcqa/mcqa_sig-whole_vocab_kl_t1_eps-1.json")
OUTPUT_PATH = INPUT_PATH.with_name("das_answer_pointer_calibration_heatmap.svg")
TARGET_VAR = "answer_pointer"


def load_heatmap_records(path: Path, target_var: str) -> tuple[list[int], list[int], dict[tuple[int, int], float]]:
    data = json.loads(path.read_text())
    payload = data["method_payloads"]["das"][0]
    records = payload["search_records"][target_var]
    layers = sorted({int(record["layer"]) for record in records})
    subspace_dims = sorted({int(record["subspace_dim"]) for record in records})
    values: dict[tuple[int, int], float] = {}
    for record in records:
        key = (int(record["layer"]), int(record["subspace_dim"]))
        values[key] = max(values.get(key, 0.0), float(record["calibration_exact_acc"]))
    return layers, subspace_dims, values


def interpolate_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    anchors = [
        (12, 44, 132),
        (44, 123, 182),
        (123, 204, 196),
        (255, 237, 160),
        (243, 119, 72),
        (165, 0, 38),
    ]
    scaled = t * (len(anchors) - 1)
    idx = int(scaled)
    frac = scaled - idx
    if idx >= len(anchors) - 1:
        r, g, b = anchors[-1]
    else:
        r0, g0, b0 = anchors[idx]
        r1, g1, b1 = anchors[idx + 1]
        r = round(r0 + frac * (r1 - r0))
        g = round(g0 + frac * (g1 - g0))
        b = round(b0 + frac * (b1 - b0))
    return f"rgb({r},{g},{b})"


def render_svg(layers: list[int], subspace_dims: list[int], values: dict[tuple[int, int], float], output_path: Path) -> None:
    width = 1120
    height = 720
    margin_left = 100
    margin_right = 120
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    cell_width = plot_width / max(len(layers), 1)
    cell_height = plot_height / max(len(subspace_dims), 1)
    all_values = list(values.values()) or [0.0]
    vmin = min(all_values)
    vmax = max(all_values)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="100" y="32" font-family="Helvetica, Arial, sans-serif" font-size="22" font-weight="700">DAS Max Calibration Accuracy Heatmap (answer_pointer)</text>',
        '<text x="100" y="54" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">Color shows max calibration exact accuracy for each (layer, subspace_dim) candidate.</text>',
    ]

    for row_index, subspace_dim in enumerate(subspace_dims):
        y = margin_top + row_index * cell_height
        for col_index, layer in enumerate(layers):
            x = margin_left + col_index * cell_width
            value = values.get((layer, subspace_dim), 0.0)
            color = interpolate_color(value, vmin, vmax)
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width:.2f}" height="{cell_height:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
            lines.append(
                f'<title>layer={layer}, subspace_dim={subspace_dim}, max_calibration_acc={value:.4f}</title>'
            )

    lines.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#111827" stroke-width="1.5"/>'
    )

    for col_index, layer in enumerate(layers):
        if layer % 2 == 0:
            x = margin_left + (col_index + 0.5) * cell_width
            lines.append(
                f'<text x="{x:.2f}" y="{height - margin_bottom + 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#111827">{layer}</text>'
            )

    for row_index, subspace_dim in enumerate(subspace_dims):
        y = margin_top + (row_index + 0.5) * cell_height
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#111827">{subspace_dim}</text>'
        )

    lines.append(
        f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#111827">Layer</text>'
    )
    lines.append(
        f'<text x="28" y="{margin_top + plot_height / 2:.2f}" transform="rotate(-90 28,{margin_top + plot_height / 2:.2f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#111827">Subspace Dimension</text>'
    )

    colorbar_x = width - margin_right + 30
    colorbar_y = margin_top
    colorbar_width = 24
    colorbar_height = plot_height
    steps = 120
    for step in range(steps):
        t0 = step / steps
        y = colorbar_y + (1.0 - t0) * colorbar_height
        color = interpolate_color(vmin + t0 * (vmax - vmin), vmin, vmax)
        lines.append(
            f'<rect x="{colorbar_x}" y="{y:.2f}" width="{colorbar_width}" height="{colorbar_height / steps + 1:.2f}" fill="{color}" stroke="none"/>'
        )
    lines.append(
        f'<rect x="{colorbar_x}" y="{colorbar_y}" width="{colorbar_width}" height="{colorbar_height}" fill="none" stroke="#111827" stroke-width="1"/>'
    )
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = vmin + tick * (vmax - vmin)
        y = colorbar_y + (1.0 - tick) * colorbar_height
        lines.append(f'<line x1="{colorbar_x + colorbar_width}" y1="{y:.2f}" x2="{colorbar_x + colorbar_width + 6}" y2="{y:.2f}" stroke="#111827" stroke-width="1"/>')
        lines.append(
            f'<text x="{colorbar_x + colorbar_width + 10}" y="{y + 4:.2f}" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#111827">{value:.3f}</text>'
        )
    lines.append(
        f'<text x="{colorbar_x + 12}" y="{colorbar_y - 12}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#111827">Max cal acc</text>'
    )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    layers, subspace_dims, values = load_heatmap_records(INPUT_PATH, TARGET_VAR)
    render_svg(layers, subspace_dims, values, OUTPUT_PATH)
    print(f"Wrote DAS calibration heatmap to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
