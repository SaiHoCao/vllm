# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


FILENAME_RE = re.compile(
    r"(?P<mode>.+)_batch(?P<batch>\d+)_seq(?P<seq>\d+)_execute_model\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot execute_model timing for split mode benchmark.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("split_mode_benchmark_results/raw_timing"),
        help="Directory containing raw_timing/<MODE>/*_execute_model.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary CSV and plots. Defaults to input-dir/plots.")
    parser.add_argument(
        "--modes",
        type=str,
        default="PADDING,DUAL_PARALLEL,DUAL_INPLACE",
        help="Comma-separated modes to include.")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Optional comma-separated seq_lens to include.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=1,
        help="Minimum matched rows required for one point.")
    parser.add_argument(
        "--include-prefill",
        action="store_true",
        help="Include non-uniform decode rows. Default only uses decode rows.")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=0,
        help="Optional maximum input batch to plot.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.")
    return parser.parse_args()


def parse_int_set(value: str) -> Optional[set[int]]:
    values = {int(x.strip()) for x in value.split(",") if x.strip()}
    return values or None


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return int(text)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def discover_files(input_dir: Path, modes: set[str]) -> list[Path]:
    paths: list[Path] = []
    for mode in sorted(modes):
        mode_dir = input_dir / mode
        if not mode_dir.exists():
            continue
        paths.extend(sorted(mode_dir.glob("*_execute_model.csv")))
    return paths


def parse_file_identity(path: Path) -> tuple[str, int, int]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected execute timing filename: {path}")
    return (match.group("mode"), int(match.group("batch")),
            int(match.group("seq")))


def row_matches_expected(row: dict[str, str], mode: str,
                         include_prefill: bool) -> bool:
    if row.get("replay_mode") != mode:
        return False
    if not include_prefill and not as_bool(row.get("uniform_decode", "False")):
        return False

    target_total = as_int(row.get("target_total"))
    actual_total = as_int(row.get("num_input_tokens"))
    if target_total is None or actual_total != target_total:
        return False

    target_first = as_int(row.get("target_first"))
    target_second = as_int(row.get("target_second"))
    should_split = target_first is not None and target_second is not None
    if row.get("split_for_cudagraph") != str(should_split):
        return False

    if not should_split:
        return True

    actual_first = as_int(row.get("first_num_tokens"))
    actual_second = as_int(row.get("second_num_tokens"))
    return actual_first == target_first and actual_second == target_second


def collect_points(input_dir: Path, modes: set[str],
                   seq_lens: Optional[set[int]], include_prefill: bool,
                   max_batch: int,
                   min_samples: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, int, int], dict[str, Any]] = {}

    for path in discover_files(input_dir, modes):
        mode, batch_size, seq_len = parse_file_identity(path)
        if seq_lens is not None and seq_len not in seq_lens:
            continue
        if max_batch and batch_size > max_batch:
            continue

        key = (mode, batch_size, seq_len)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not row_matches_expected(row, mode, include_prefill):
                    continue
                grouped[key].append(float(row["elapsed_ms"]))
                metadata[key] = {
                    "target_total": as_int(row.get("target_total")),
                    "target_first": as_int(row.get("target_first")),
                    "target_second": as_int(row.get("target_second")),
                }

    points: list[dict[str, Any]] = []
    for (mode, batch_size, seq_len), times in sorted(grouped.items()):
        if len(times) < min_samples:
            continue
        target_total = metadata[(mode, batch_size, seq_len)]["target_total"]
        padding_tokens = None
        padding_ratio = None
        if target_total is not None:
            padding_tokens = max(0, int(target_total) - int(batch_size))
            padding_ratio = padding_tokens / max(1, int(target_total))
        points.append({
            "mode": mode,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "padding_tokens": padding_tokens,
            "padding_ratio": padding_ratio,
            "count": len(times),
            "mean_ms": statistics.mean(times),
            "median_ms": statistics.median(times),
            "min_ms": min(times),
            "max_ms": max(times),
            **metadata[(mode, batch_size, seq_len)],
        })
    return points


def write_summary(points: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "execute_model_matched_summary.csv"
    fieldnames = [
        "mode", "batch_size", "seq_len", "target_total", "target_first",
        "target_second", "padding_tokens", "padding_ratio", "count",
        "mean_ms", "median_ms", "min_ms", "max_ms"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    return path


def plot_points(points: list[dict[str, Any]], output_dir: Path,
                show: bool) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return []

    output_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lens = sorted({int(p["seq_len"]) for p in points})
    modes = sorted({str(p["mode"]) for p in points})
    colors = {
        "PADDING": "#4C78A8",
        "DUAL_PARALLEL": "#F58518",
        "DUAL_INPLACE": "#54A24B",
    }
    markers = {
        128: "P",
        256: "X",
        512: "o",
        1024: "s",
        2048: "^",
        4096: "D",
        8192: "v",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for seq_len in seq_lens:
        for mode in modes:
            rows = [
                p for p in points
                if p["mode"] == mode and int(p["seq_len"]) == seq_len
            ]
            if not rows:
                continue
            rows.sort(key=lambda p: int(p["batch_size"]))
            ax.plot(
                [int(p["batch_size"]) for p in rows],
                [float(p["mean_ms"]) for p in rows],
                marker=markers.get(seq_len, "o"),
                linewidth=2,
                markersize=5,
                color=colors.get(mode),
                label=f"{mode} seq={seq_len}",
            )
    ax.set_xlabel("Input batch size")
    ax.set_ylabel("execute_model mean time (ms)")
    ax.set_title("Matched decode execute_model timing")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    combined_path = output_dir / "execute_model_matched_by_batch.png"
    fig.savefig(combined_path, dpi=180)
    output_paths.append(combined_path)

    for seq_len in seq_lens:
        fig, ax = plt.subplots(figsize=(10, 6))
        pad_ax = ax.twinx()
        batch_values = sorted({
            int(p["batch_size"])
            for p in points if int(p["seq_len"]) == seq_len
        })
        if len(batch_values) > 1:
            min_gap = min(b - a for a, b in zip(batch_values, batch_values[1:]))
            bar_width = max(1.0, min_gap * 0.18)
        else:
            bar_width = max(1.0, batch_values[0] * 0.02) if batch_values else 1.0
        mode_count = max(1, len(modes))
        mode_offsets = {
            mode: (idx - (mode_count - 1) / 2) * bar_width
            for idx, mode in enumerate(modes)
        }
        bar_handles = []
        for mode in modes:
            rows = [
                p for p in points
                if p["mode"] == mode and int(p["seq_len"]) == seq_len
            ]
            if not rows:
                continue
            rows.sort(key=lambda p: int(p["batch_size"]))
            ax.plot(
                [int(p["batch_size"]) for p in rows],
                [float(p["mean_ms"]) for p in rows],
                marker="o",
                linewidth=2,
                markersize=5,
                color=colors.get(mode),
                label=mode,
            )
            bar = pad_ax.bar(
                [
                    int(p["batch_size"]) + mode_offsets[mode]
                    for p in rows
                ],
                [int(p["padding_tokens"] or 0) for p in rows],
                width=bar_width,
                color=colors.get(mode),
                alpha=0.18,
                edgecolor="none",
                label=f"{mode} padding",
            )
            bar_handles.append(bar)
        ax.set_xlabel("Input batch size")
        ax.set_ylabel("execute_model mean time (ms)")
        pad_ax.set_ylabel("padding tokens")
        pad_ax.set_ylim(bottom=0)
        ax.set_title(f"Matched decode execute_model timing, seq={seq_len}")
        ax.grid(True, alpha=0.3)
        line_handles, line_labels = ax.get_legend_handles_labels()
        bar_handles, bar_labels = pad_ax.get_legend_handles_labels()
        ax.legend(line_handles + bar_handles,
                  line_labels + bar_labels,
                  fontsize=9,
                  ncol=2)
        fig.tight_layout()
        path = output_dir / f"execute_model_matched_seq{seq_len}.png"
        fig.savefig(path, dpi=180)
        output_paths.append(path)

    if show:
        plt.show()
    else:
        plt.close("all")
    return output_paths


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "plots"
    modes = {x.strip() for x in args.modes.split(",") if x.strip()}
    seq_lens = parse_int_set(args.seq_lens)

    points = collect_points(input_dir=input_dir,
                            modes=modes,
                            seq_lens=seq_lens,
                            include_prefill=args.include_prefill,
                            max_batch=args.max_batch,
                            min_samples=args.min_samples)
    summary_path = write_summary(points, output_dir)
    plot_paths = plot_points(points, output_dir, args.show)

    print(f"matched_points={len(points)}")
    print(f"summary={summary_path}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
