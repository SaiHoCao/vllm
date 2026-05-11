# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

# 画时间，头开销比例，和baseline的优化比
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot split mode summary without rerunning benchmark.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Benchmark output dir containing split_mode_execute_summary.csv.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Explicit summary CSV or JSON path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot output dir. Defaults to <input-dir>/plots_from_summary.")
    parser.add_argument(
        "--modes",
        type=str,
        default="",
        help="Optional comma-separated modes to include, e.g. EAGER,PADDING.")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Optional comma-separated seq_lens to include.")
    parser.add_argument(
        "--stat",
        choices=("mean", "p50", "p90", "p99"),
        default="mean",
        help="Statistic to plot for total/phase/head overhead.")
    parser.add_argument(
        "--baseline",
        type=str,
        default="EAGER",
        help="Baseline mode for speedup plot. Default: EAGER.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="Model name shown in plot titles. If empty, infer from "
        "--input-dir names like qwen3_4b or qwen3_0.6b.")
    parser.add_argument(
        "--title-suffix",
        type=str,
        default="",
        help="Extra title text appended after inferred model and seq range.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.")
    return parser.parse_args()


def parse_int_set(value: str) -> Optional[set[int]]:
    values = {int(x.strip()) for x in value.split(",") if x.strip()}
    return values or None


def parse_str_set(value: str) -> Optional[set[str]]:
    values = {x.strip().upper() for x in value.split(",") if x.strip()}
    return values or None


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def as_int(value: Any) -> Optional[int]:
    number = as_float(value)
    return int(number) if number is not None else None


def resolve_summary_path(args: argparse.Namespace) -> Path:
    if args.summary is not None:
        return args.summary
    if args.input_dir is None:
        raise ValueError("Pass --input-dir or --summary.")
    return args.input_dir / "split_mode_execute_summary.csv"


def infer_model_name(input_dir: Optional[Path]) -> str:
    if input_dir is None:
        return ""
    name = input_dir.name.lower()
    if "qwen3_4b" in name:
        return "Qwen3-4B"
    if "qwen3_0.6b" in name or "qwen3_0_6b" in name:
        return "Qwen3-0.6B"
    return ""


def infer_seq_range_label(input_dir: Optional[Path]) -> str:
    if input_dir is None:
        return ""
    name = input_dir.name.lower()
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?k)_(\d+(?:\.\d+)?k)"
                      r"(?![a-z0-9])", name)
    if not match:
        return ""
    if match.group(1) == match.group(2):
        return f"seq_len {match.group(1)}"
        
    return f"seq_len {match.group(1)}-{match.group(2)}"


def load_summary(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        return [dict(row) for row in data]

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_name(prefix: str, stat: str) -> str:
    if prefix == "total":
        return {
            "mean": "gpu_mean_ms",
            "p50": "gpu_p50_ms",
            "p90": "p90_ms",
            "p99": "gpu_p99_ms",
        }[stat]
    return {
        "mean": f"{prefix}_mean_ms",
        "p50": f"{prefix}_p50_ms",
        "p90": f"{prefix}_p90_ms",
        "p99": f"{prefix}_p99_ms",
    }[stat]


def normalize_rows(rows: list[dict[str, Any]], stat: str,
                   modes: Optional[set[str]],
                   seq_lens: Optional[set[int]]) -> list[dict[str, Any]]:
    total_key = metric_name("total", stat)
    phase_key = metric_name("phase", stat)
    head_key = metric_name("head_overhead", stat)

    points: list[dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("mode", "")).upper()
        seq_len = as_int(row.get("seq_len"))
        batch_size = as_int(row.get("batch_size"))
        if not mode or seq_len is None or batch_size is None:
            continue
        if modes is not None and mode not in modes:
            continue
        if seq_lens is not None and seq_len not in seq_lens:
            continue

        total_ms = as_float(row.get(total_key))
        if total_ms is None:
            total_ms = as_float(row.get("mean_ms"))
        phase_ms = as_float(row.get(phase_key))
        head_ms = as_float(row.get(head_key))
        if head_ms is None and total_ms is not None and phase_ms is not None:
            head_ms = total_ms - phase_ms
        ratio = None
        if total_ms is not None and head_ms is not None and total_ms > 0:
            ratio = head_ms / total_ms * 100

        points.append({
            "mode": mode,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "total_ms": total_ms,
            "phase_ms": phase_ms,
            "head_ms": head_ms,
            "head_ratio_pct": ratio,
        })
    return points


def plot_summary(points: list[dict[str, Any]], output_dir: Path, stat: str,
                 baseline: str, model_name: str, title_suffix: str,
                 show: bool) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    modes = sorted({p["mode"] for p in points})
    colors = {
        "EAGER": "#4C78A8",
        "PADDING": "#F58518",
        "DUAL_PARALLEL": "#54A24B",
        "DUAL_INPLACE": "#B279A2",
        "DUAL_MIXED": "#72B7B2",
    }
    markers = {
        "EAGER": "o",
        "PADDING": "s",
        "DUAL_PARALLEL": "^",
        "DUAL_INPLACE": "D",
        "DUAL_MIXED": "X",
    }

    def title(main: str) -> str:
        parts = [main]
        if model_name:
            parts.append(model_name)
        if title_suffix:
            parts.append(title_suffix)
        return " | ".join(parts)

    output_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for mode in modes:
        rows = sorted([p for p in points if p["mode"] == mode],
                      key=lambda p: (p["batch_size"], p["seq_len"]))
        rows_total = [p for p in rows if p["total_ms"] is not None]
        rows_head = [p for p in rows if p["head_ms"] is not None]
        if rows_total:
            ax.plot([p["batch_size"] for p in rows_total],
                    [p["total_ms"] for p in rows_total],
                    marker=markers.get(mode, "o"),
                    linewidth=2,
                    markersize=5.5,
                    color=colors.get(mode),
                    label=f"{mode} total")
        if rows_head:
            ax.plot([p["batch_size"] for p in rows_head],
                    [p["head_ms"] for p in rows_head],
                    marker=markers.get(mode, "o"),
                    linewidth=2,
                    linestyle="--",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgewidth=1.3,
                    color=colors.get(mode),
                    label=f"{mode} head")
    ax.set_title(title(f"Decode execute total and head overhead ({stat})"))
    ax.set_xlabel("batch size")
    ax.set_ylabel("time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path = output_dir / f"batch_vs_total_and_head_{stat}.png"
    fig.savefig(path, dpi=180)
    output_paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for mode in modes:
        rows = sorted([
            p for p in points
            if p["mode"] == mode and p["head_ratio_pct"] is not None
        ], key=lambda p: (p["batch_size"], p["seq_len"]))
        if not rows:
            continue
        ax.plot([p["batch_size"] for p in rows],
                [p["head_ratio_pct"] for p in rows],
                marker=markers.get(mode, "o"),
                linewidth=2,
                markersize=5.5,
                color=colors.get(mode),
                label=mode)
    ax.set_title(title(f"Head overhead ratio ({stat})"))
    ax.set_xlabel("batch size")
    ax.set_ylabel("head overhead / execute total (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / f"batch_vs_head_overhead_ratio_{stat}.png"
    fig.savefig(path, dpi=180)
    output_paths.append(path)
    plt.close(fig)

    baseline = baseline.upper()
    baseline_by_key = {
        (p["batch_size"], p["seq_len"]): p["total_ms"]
        for p in points
        if p["mode"] == baseline and p["total_ms"] is not None
    }
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axhline(0, color="#666666", linewidth=1, alpha=0.7)
    for mode in modes:
        if mode == baseline:
            continue
        rows = []
        for p in points:
            if p["mode"] != mode or p["total_ms"] is None:
                continue
            base = baseline_by_key.get((p["batch_size"], p["seq_len"]))
            if base is None or base <= 0:
                continue
            rows.append({
                **p,
                "speedup_pct": (base - p["total_ms"]) / base * 100,
            })
        rows.sort(key=lambda p: (p["batch_size"], p["seq_len"]))
        if not rows:
            continue
        ax.plot([p["batch_size"] for p in rows],
                [p["speedup_pct"] for p in rows],
                marker=markers.get(mode, "o"),
                linewidth=2,
                markersize=5.5,
                color=colors.get(mode),
                label=f"{mode} vs {baseline}")
    ax.set_title(title(f"Decode total speedup vs {baseline} ({stat})"))
    ax.set_xlabel("batch size")
    ax.set_ylabel("speedup vs baseline (%), higher is faster")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = output_dir / f"batch_vs_{baseline.lower()}_speedup_{stat}.png"
    fig.savefig(path, dpi=180)
    output_paths.append(path)
    plt.close(fig)

    if show:
        plt.show()
    return output_paths


def main() -> None:
    args = parse_args()
    summary_path = resolve_summary_path(args)
    base_output_dir = args.output_dir
    if base_output_dir is None:
        base_output_dir = summary_path.parent / "plots_from_summary"

    rows = load_summary(summary_path)
    points = normalize_rows(rows, args.stat, parse_str_set(args.modes),
                            parse_int_set(args.seq_lens))
    model_name = args.model_name or infer_model_name(args.input_dir)
    inferred_seq_range = infer_seq_range_label(args.input_dir)
    title_parts = [part for part in (inferred_seq_range, args.title_suffix)
                   if part]
    title_suffix = " | ".join(title_parts)
    plot_paths = plot_summary(points, base_output_dir, args.stat,
                              args.baseline, model_name, title_suffix,
                              args.show)

    print(f"summary={summary_path}")
    print(f"points={len(points)}")
    if model_name:
        print(f"model_name={model_name}")
    if inferred_seq_range:
        print(f"seq_range={inferred_seq_range}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
