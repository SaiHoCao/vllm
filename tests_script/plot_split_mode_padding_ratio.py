# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


# Padding max 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot split mode padding-ratio benchmark summaries.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("split_mode_benchmark_results"),
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
        help="Plot output dir. Defaults to <input-dir>/padding_ratio_plots.")
    parser.add_argument(
        "--modes",
        type=str,
        default="PADDING_MAX,PADDING,DUAL_INPLACE,DUAL_PARALLEL",
        help="Comma-separated modes to include.")
    parser.add_argument(
        "--metrics",
        type=str,
        default="total,forward",
        help="Comma-separated metrics: total,forward,overhead.")
    parser.add_argument(
        "--stat",
        choices=("mean", "p50", "p90", "p99"),
        default="mean",
        help="Statistic to plot.")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Optional comma-separated seq_lens to include.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum matched execute rows required for a point.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.")
    parser.add_argument(
        "--rebuild-summary-from-raw",
        action="store_true",
        help="Ignore split_mode_execute_summary and rebuild plot points from "
        "raw_timing/<MODE>/*_execute_model.csv.")
    return parser.parse_args()


def parse_str_set(value: str) -> Optional[set[str]]:
    values = {x.strip().upper() for x in value.split(",") if x.strip()}
    return values or None


def parse_int_set(value: str) -> Optional[set[int]]:
    values = {int(x.strip()) for x in value.split(",") if x.strip()}
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
    return args.input_dir / "split_mode_execute_summary.csv"


def load_summary(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        return [dict(row) for row in data]

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_times(times: list[float]) -> dict[str, Optional[float]]:
    if not times:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    values = sorted(times)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": statistics.median(values),
        "p90": values[round((len(values) - 1) * 0.9)],
        "p99": values[round((len(values) - 1) * 0.99)],
    }


def row_matches_target(row: dict[str, str]) -> bool:
    if str(row.get("uniform_decode", "True")) != "True":
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
    return (as_int(row.get("first_num_tokens")) == target_first
            and as_int(row.get("second_num_tokens")) == target_second)


def rebuild_summary_from_raw(input_dir: Path,
                             modes: Optional[set[str]]) -> list[dict[str, Any]]:
    raw_dir = input_dir / "raw_timing"
    grouped: dict[tuple[str, int, int], dict[str, Any]] = {}
    total_times: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    phase_times: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    overhead_times: dict[tuple[str, int, int], list[float]] = defaultdict(list)

    for mode_dir in sorted(raw_dir.glob("*")):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name.upper()
        if modes is not None and mode not in modes:
            continue
        for path in sorted(mode_dir.glob("*_execute_model.csv")):
            with path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row_matches_target(row):
                        continue
                    batch_size = as_int(row.get("input_batch_size"))
                    seq_len = as_int(row.get("seq_len"))
                    target_total = as_int(row.get("target_total"))
                    if batch_size is None or seq_len is None:
                        continue
                    key = (mode, batch_size, seq_len)
                    grouped[key] = {
                        "mode": mode,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "expected_total": target_total,
                        "expected_first": as_int(row.get("target_first")),
                        "expected_second": as_int(row.get("target_second")),
                        "padding_tokens": max(0,
                                              (target_total or batch_size)
                                              - batch_size),
                        "scheme": str(target_total or ""),
                    }
                    if grouped[key]["expected_first"] is not None:
                        grouped[key]["scheme"] = (
                            f"{grouped[key]['expected_first']}+"
                            f"{grouped[key]['expected_second']}")
                    total = as_float(row.get("gpu_elapsed_ms"))
                    if total is None:
                        total = as_float(row.get("elapsed_ms"))
                    phase = as_float(row.get("phase_gpu_elapsed_ms"))
                    overhead = as_float(row.get("head_overhead_ms"))
                    if total is not None:
                        total_times[key].append(total)
                    if phase is not None:
                        phase_times[key].append(phase)
                    if overhead is not None:
                        overhead_times[key].append(overhead)

    rows: list[dict[str, Any]] = []
    for key, row in sorted(grouped.items()):
        total = summarize_times(total_times[key])
        phase = summarize_times(phase_times[key])
        overhead = summarize_times(overhead_times[key])
        expected_total = row.get("expected_total") or row["batch_size"]
        padding_tokens = row["padding_tokens"]
        rows.append({
            **row,
            "padding_ratio": padding_tokens / max(1, expected_total),
            "count": total["count"],
            "gpu_mean_ms": total["mean"],
            "gpu_p50_ms": total["p50"],
            "gpu_p90_ms": total["p90"],
            "gpu_p99_ms": total["p99"],
            "phase_mean_ms": phase["mean"],
            "phase_p50_ms": phase["p50"],
            "phase_p90_ms": phase["p90"],
            "phase_p99_ms": phase["p99"],
            "head_overhead_mean_ms": overhead["mean"],
            "head_overhead_p50_ms": overhead["p50"],
            "head_overhead_p90_ms": overhead["p90"],
            "head_overhead_p99_ms": overhead["p99"],
        })
    return rows


def metric_column(metric: str, stat: str) -> str:
    columns = {
        "total": {
            "mean": "gpu_mean_ms",
            "p50": "gpu_p50_ms",
            "p90": "gpu_p90_ms",
            "p99": "gpu_p99_ms",
        },
        "forward": {
            "mean": "phase_mean_ms",
            "p50": "phase_p50_ms",
            "p90": "phase_p90_ms",
            "p99": "phase_p99_ms",
        },
        "overhead": {
            "mean": "head_overhead_mean_ms",
            "p50": "head_overhead_p50_ms",
            "p90": "head_overhead_p90_ms",
            "p99": "head_overhead_p99_ms",
        },
    }
    return columns[metric][stat]


def normalize_rows(rows: list[dict[str, Any]], modes: Optional[set[str]],
                   seq_lens: Optional[set[int]], min_count: int,
                   metrics: list[str], stat: str) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("mode", "")).upper()
        batch_size = as_int(row.get("batch_size"))
        seq_len = as_int(row.get("seq_len"))
        count = as_int(row.get("count")) or 0
        if not mode or batch_size is None or seq_len is None:
            continue
        if modes is not None and mode not in modes:
            continue
        if seq_lens is not None and seq_len not in seq_lens:
            continue
        if count < min_count:
            continue

        point = {
            "mode": mode,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "count": count,
            "padding_tokens": as_int(row.get("padding_tokens")),
            "padding_ratio": as_float(row.get("padding_ratio")),
            "scheme": row.get("scheme", ""),
        }
        for metric in metrics:
            value = as_float(row.get(metric_column(metric, stat)))
            if metric == "total" and value is None:
                value = as_float(row.get("mean_ms"))
            point[f"{metric}_ms"] = value
        points.append(point)
    return points


def write_points(points: list[dict[str, Any]], output_dir: Path,
                 metrics: list[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "padding_ratio_plot_points.csv"
    fieldnames = [
        "mode", "batch_size", "seq_len", "count", "padding_tokens",
        "padding_ratio", "scheme"
    ] + [f"{metric}_ms" for metric in metrics]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in points:
            writer.writerow({key: point.get(key) for key in fieldnames})
    return path


def plot_points(points: list[dict[str, Any]], output_dir: Path,
                metrics: list[str], stat: str, show: bool) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    modes = sorted({p["mode"] for p in points})
    seq_lens = sorted({p["seq_len"] for p in points})
    colors = {
        "PADDING_MAX": "#1F77B4",
        "PADDING": "#FF7F0E",
        "DUAL_INPLACE": "#8BF905",
        "DUAL_PARALLEL": "#D62728",
        "EAGER": "#042A76",
    }
    markers = {
        "PADDING_MAX": "o",
        "PADDING": "s",
        "DUAL_INPLACE": "D",
        "DUAL_PARALLEL": "^",
        "EAGER": "X",
    }
    metric_labels = {
        "total": "execute_model total GPU time",
        "forward": "matched forward/replay GPU time",
        "overhead": "execute_model overhead GPU time",
    }
    output_paths: list[Path] = []

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for mode in modes:
            rows = sorted([
                p for p in points
                if p["mode"] == mode and p.get(f"{metric}_ms") is not None
            ], key=lambda p: (p["batch_size"], p["seq_len"]))
            if not rows:
                continue
            ax.plot([p["batch_size"] for p in rows],
                    [p[f"{metric}_ms"] for p in rows],
                    marker=markers.get(mode, "o"),
                    linewidth=2,
                    markersize=5.5,
                    color=colors.get(mode),
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=mode)
        ax.set_title(f"{metric_labels[metric]} ({stat})")
        ax.set_xlabel("input batch size")
        ax.set_ylabel("time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9,
                  frameon=True,
                  fancybox=False,
                  framealpha=0.95,
                  handlelength=2.2,
                  markerscale=1.25)
        fig.tight_layout()
        path = output_dir / f"batch_vs_{metric}_{stat}.png"
        fig.savefig(path, dpi=180)
        output_paths.append(path)
        plt.close(fig)

    for seq_len in seq_lens:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for metric in metrics:
            for mode in modes:
                rows = sorted([
                    p for p in points
                    if p["mode"] == mode and p["seq_len"] == seq_len
                    and p.get(f"{metric}_ms") is not None
                ], key=lambda p: p["batch_size"])
                if not rows:
                    continue
                linestyle = "-" if metric == "total" else "--"
                if metric == "overhead":
                    linestyle = ":"
                ax.plot([p["batch_size"] for p in rows],
                        [p[f"{metric}_ms"] for p in rows],
                        marker=markers.get(mode, "o"),
                        linewidth=2,
                        linestyle=linestyle,
                        markersize=5.5,
                        color=colors.get(mode),
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        label=f"{mode} {metric}")
        ax.set_title(f"split mode timing, seq_len={seq_len} ({stat})")
        ax.set_xlabel("input batch size")
        ax.set_ylabel("time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2,
                  fontsize=8.5,
                  frameon=True,
                  fancybox=False,
                  framealpha=0.95,
                  handlelength=2.4,
                  markerscale=1.2)
        fig.tight_layout()
        path = output_dir / f"batch_vs_selected_metrics_seq{seq_len}_{stat}.png"
        fig.savefig(path, dpi=180)
        output_paths.append(path)
        plt.close(fig)

    if show:
        plt.show()
    return output_paths


def main() -> None:
    args = parse_args()
    metrics = [x.strip().lower() for x in args.metrics.split(",")
               if x.strip()]
    bad_metrics = sorted(set(metrics) - {"total", "forward", "overhead"})
    if bad_metrics:
        raise ValueError(f"Unsupported metrics: {bad_metrics}")

    output_dir = args.output_dir or args.input_dir / "padding_ratio_plots"
    modes = parse_str_set(args.modes)
    if args.rebuild_summary_from_raw:
        rows = rebuild_summary_from_raw(args.input_dir, modes)
    else:
        rows = load_summary(resolve_summary_path(args))
    points = normalize_rows(rows=rows,
                            modes=modes,
                            seq_lens=parse_int_set(args.seq_lens),
                            min_count=args.min_count,
                            metrics=metrics,
                            stat=args.stat)
    summary_path = write_points(points, output_dir, metrics)
    plot_paths = plot_points(points, output_dir, metrics, args.stat, args.show)

    print(f"points={len(points)}")
    print(f"summary={summary_path}")
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
