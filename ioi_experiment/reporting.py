"""Summary helpers for IOI comparison runs."""

from __future__ import annotations

from collections import defaultdict

from .runtime import ensure_parent_dir


SPLIT_PRINT_ORDER = ("train", "calibration", "test")


def summarize_method_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["method"])].append(record)
    summaries = []
    for method, method_records in sorted(grouped.items()):
        exact_acc = sum(float(record["exact_acc"]) for record in method_records) / len(method_records)
        choice_acc = sum(float(record.get("choice_acc", record["exact_acc"])) for record in method_records) / len(method_records)
        summaries.append({"method": method, "exact_acc": exact_acc, "choice_acc": choice_acc})
    return summaries


def write_text_report(path, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def format_summary(
    *,
    model_name: str,
    data_metadata: dict[str, object],
    method_payloads: dict[str, list[dict[str, object]]],
    summary_records: list[dict[str, object]],
) -> str:
    lines = [
        "IOI Compare Summary",
        f"model: {model_name}",
        "",
        "data:",
    ]
    for split in SPLIT_PRINT_ORDER:
        if split in data_metadata:
            lines.append(f"{split}: {data_metadata[split]}")
    lines.append("")
    lines.append("pair bank summary:")
    for split in SPLIT_PRINT_ORDER:
        split_metadata = data_metadata.get(split)
        if isinstance(split_metadata, dict) and split_metadata:
            sample_stats = next(iter(split_metadata.values()))
            lines.append(
                f"{split} pair bank | total_pairs={int(sample_stats.get('size', 0))} "
                f"| datasets={sample_stats.get('dataset_names', [])}"
            )
            for variable, variable_stats in sorted(split_metadata.items()):
                total_pairs = int(variable_stats.get("size", 0))
                changed_count = int(variable_stats.get("changed_count", 0))
                unchanged_count = max(0, total_pairs - changed_count)
                lines.append(
                    f"{split} pair bank [{variable}] | changed={changed_count} "
                    f"| unchanged={unchanged_count} "
                    f"| changed_rate={float(variable_stats.get('changed_rate', 0.0)):.4f}"
                )
    lines.append("")
    lines.append("method summary:")
    for record in summary_records:
        lines.append(
            f"{str(record['method']).upper()}: choice={float(record.get('choice_acc', record['exact_acc'])):.4f}"
            f"  exact(full-vocab)={float(record['exact_acc']):.4f}"
        )
    lines.append("")
    lines.append("best selections:")
    for method, payload_list in sorted(method_payloads.items()):
        for payload in payload_list:
            for record in payload.get("results", []):
                bits = [
                    f"{method.upper()}[{record['variable']}]",
                    f"choice={float(record.get('choice_acc', record['exact_acc'])):.4f}",
                    f"exact={float(record['exact_acc']):.4f}",
                    f"site={record.get('site_label')}",
                ]
                if "token_position_id" in record:
                    bits.append(f"token_position={record['token_position_id']}")
                if "subspace_dim" in record:
                    bits.append(f"subspace_dim={record['subspace_dim']}")
                if "top_k" in record:
                    bits.append(f"top_k={record['top_k']}")
                if "lambda" in record:
                    bits.append(f"lambda={record['lambda']}")
                if "signature_mode" in record:
                    bits.append(f"signature_mode={record['signature_mode']}")
                lines.append(", ".join(bits))
    return "\n".join(lines)


def print_results_table(records: list[dict[str, object]], title: str) -> None:
    print(title)
    if not records:
        print("(no records)")
        return
    header = f"{'method':<8} {'variable':<16} {'choice':>8} {'exact':>8} {'select/cal':>10} {'site/config':<32}"
    print(header)
    print("-" * len(header))
    for record in records:
        site_bits = [str(record.get("site_label", "n/a"))]
        if "subspace_dim" in record:
            site_bits.append(f"k={record['subspace_dim']}")
        if "top_k" in record:
            site_bits.append(f"topk={record['top_k']}")
        if "lambda" in record:
            site_bits.append(f"l={record['lambda']}")
        if "signature_mode" in record:
            site_bits.append(str(record["signature_mode"]))
        print(
            f"{str(record['method']):<8} "
            f"{str(record.get('variable', 'average')):<16} "
            f"{float(record.get('choice_acc', record['exact_acc'])):>8.4f} "
            f"{float(record['exact_acc']):>8.4f} "
            f"{float(record.get('selection_exact_acc', record.get('calibration_exact_acc', 0.0))):>10.4f} "
            f"{' '.join(site_bits):<32}"
        )
