# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import csv
import gc
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_from_disk

from vllm import EngineArgs, LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.utils import FlexibleArgumentParser

# 用于多个模式，多个测试用例测试
# example:
# python tests_script/split_mode_benchmark.py   
# --model /home/csh/data/Qwen3-4B   
# --dataset-path /home/csh/data/projects/datasets/LongBench-v2   
# --modes PADDING,EAGER,DUAL_PARALLEL,DUAL_INPLACE   
# --experiment1-batches   
# --random-seq-len-range 100,150   --max-tokens 256   --repeat 1   
# --output-dir split_mode_test
# --gpu-memory-utilization 0.6

REPLAY_MODES = [
    "EAGER", "PADDING_MAX", "PADDING", "DUAL_PARALLEL", "DUAL_INPLACE"
]
PADDING_RATIO_512_CAPTURE_PRESET = {
    "PADDING_MAX": [512],
    "PADDING": [
        1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288,
        320, 352, 384, 416, 448, 480, 512
    ],
    "DUAL_INPLACE": [
        1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512
    ],
    "DUAL_PARALLEL": [
        1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288,
        # Skip 416 for now: batch=432 would split into 416+16, which is
        # currently unstable in the dual parallel two-stream replay path.
        320, 352, 384, 416, 448, 480, 512
    ],
}
PADDING_RATIO_512_BATCH_SIZES = list(range(272, 512, 16))

EXPERIMENT1_BATCH_SIZES = [
    # 32, 40, 48, 60,
    # 64, 72, 96, 120,
    128, 144, 192, 240,
#     256, 272, 320, 368,
#     384, 400, 448, 496,
]


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    # cudagraph_sizes = [1, 2, 4, 8, 16, 32, 64] + [
    #     i * 128 for i in range(1, 6)
    # ] + [896]
    cudagraph_sizes = [1, 2, 4, 8, 16, 32, 64] + [
        i * 128 for i in range(1, 5)
    ]
    parser.set_defaults(
        compilation_config={
            "level": "3",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": cudagraph_sizes,
            "replay_mode": "PADDING",
        })
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=16384)
    parser.set_defaults(enable_chunked_prefill=False)
    parser.set_defaults(enable_prefix_caching=False)
    parser.set_defaults(disable_log_stats=True)

    group = parser.add_argument_group("Split benchmark")
    group.add_argument("--dataset-path",
                       type=str,
                       default="/home/csh/data/projects/datasets/LongBench-v2")
    group.add_argument("--modes",
                       type=str,
                       default="EAGER,DUAL_PARALLEL")
    group.add_argument("--capture-size-preset",
                       choices=("shared", "padding-ratio-512"),
                       default="shared",
                       help="Use per-mode capture sizes. 'shared' preserves "
                       "the original behavior. 'padding-ratio-512' uses the "
                       "four configs in this benchmark: PADDING_MAX=[512], "
                       "normal PADDING every 32 from 64 to 512, "
                       "DUAL_INPLACE every 64 from 64 to 512, and "
                       "DUAL_PARALLEL every 32 from 64 to 512.")
    group.add_argument("--batch-sizes",
                       type=str,
                       default="",
                       help="Comma-separated batch sizes. If empty, generate "
                       "anchors around cudagraph buckets.")
    group.add_argument("--experiment1-batches",
                       action="store_true",
                       default=True,
                       help="Use the 20 batch sizes from experiment 1.")
    group.add_argument("--no-experiment1-batches",
                       dest="experiment1_batches",
                       action="store_false")
    group.add_argument("--padding-ratio-batches",
                       action="store_true",
                       help="Use batches 272,288,...,496, i.e. "
                       "256 + (1/16..15/16) * (512 - 256).")
    group.add_argument("--ratio-batches",
                       action="store_true",
                       help="Use ratio-spaced batches between adjacent "
                       "cudagraph buckets. With the default "
                       "--batch-ratio-parts=4, this is the 1/4ratio group.")
    group.add_argument("--batch-ratio-parts",
                       type=int,
                       default=4,
                       help="Split each adjacent cudagraph bucket gap into "
                       "this many parts for --ratio-batches.")
    group.add_argument("--min-batch-size", type=int, default=256)
    group.add_argument("--max-batch-size", type=int, default=640)
    group.add_argument("--seq-lens", type=str, default="512,1024,2048")
    group.add_argument("--random-seq-len-range",
                       type=str,
                       default="",
                       help="Generate one deterministic random seq_len per "
                       "batch, e.g. 1000,2000.")
    group.add_argument("--max-kv-tokens",
                       type=int,
                       default=2097152,
                       help="Skip batch*seq_len above this value.")
    group.add_argument("--max-tokens",
                       type=int,
                       default=256,
                       help="Generated decode tokens per request.")
    group.add_argument("--input-mode",
                       choices=("tokens", "text"),
                       default="tokens",
                       help="tokens passes exact prompt_token_ids of length "
                       "seq_len; text keeps normal vLLM tokenization.")
    group.add_argument("--repeat", type=int, default=1)
    group.add_argument("--seed-r", type=int, default=42)
    group.add_argument("--output-dir",
                       type=str,
                       default="split_mode_benchmark_results")
    group.add_argument("--micro-batch-size",
                       type=int,
                       default=None,
                       help="Override GPUModelRunner.micro_batch_size. "
                       "Defaults to max adjacent cudagraph bucket gap.")
    group.add_argument("--cudagraph-split-pad-threshold",
                       type=int,
                       default=0,
                       help="Must match compilation_config."
                       "cudagraph_split_pad_threshold.")
    return parser


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_int_range(value: str) -> tuple[int, int]:
    values = parse_int_list(value)
    if len(values) != 2:
        raise ValueError(f"Expected range like '1000,2000', got {value!r}")
    start, end = values
    if start > end:
        start, end = end, start
    return start, end


def generate_stratified_seq_lens(batch_sizes: list[int], seq_min: int,
                                 seq_max: int, seed: int) -> dict[int, int]:
    rng = random.Random(seed)
    if len(batch_sizes) == 1:
        return {batch_sizes[0]: rng.randint(seq_min, seq_max)}

    width = seq_max - seq_min + 1
    seq_by_batch: dict[int, int] = {}
    sorted_batches = sorted(batch_sizes)
    for idx, batch_size in enumerate(sorted_batches):
        reverse_idx = len(sorted_batches) - 1 - idx
        cell_start = seq_min + int(reverse_idx * width / len(batch_sizes))
        cell_end = seq_min + int((reverse_idx + 1) * width /
                                  len(batch_sizes)) - 1
        cell_end = max(cell_start, min(seq_max, cell_end))
        seq_by_batch[batch_size] = rng.randint(cell_start, cell_end)
    return seq_by_batch


def timing_mode_name(mode: str) -> str:
    if mode == "EAGER":
        return "NONE"
    if mode == "PADDING_MAX":
        return "PADDING"
    return mode


def replay_mode_name(mode: str) -> str:
    return "PADDING" if mode == "PADDING_MAX" else mode


def capture_sizes_for_mode(mode: str, base_buckets: list[int],
                           preset: str) -> list[int]:
    if preset == "padding-ratio-512":
        return sorted(
            set(PADDING_RATIO_512_CAPTURE_PRESET.get(mode, base_buckets)))
    return sorted(set(base_buckets))


def truncate_or_pad_text(text: str,
                         target_tokens: int,
                         avg_chars_per_token: float = 4.0) -> str:
    target_chars = max(1, int(target_tokens * avg_chars_per_token))
    if len(text) >= target_chars:
        return text[:target_chars]
    repeated = text * (target_chars // max(1, len(text)) + 1)
    return repeated[:target_chars]


def get_sample_text(sample: dict[str, Any]) -> str:
    for key in ("context", "input", "question"):
        if key in sample:
            return str(sample[key])
    return str(list(sample.values())[0])


def prepare_prompts(dataset: Optional[Any], batch_size: int,
                    seq_len: int) -> list[str]:
    if dataset is None:
        base_text = "This is a split benchmark prompt. " * max(1, seq_len // 6)
        return [f"{base_text}\n\nSummarize:" for _ in range(batch_size)]

    prompts: list[str] = []
    for _ in range(batch_size):
        sample = dataset[random.randint(0, len(dataset) - 1)]
        text = get_sample_text(sample)
        prompts.append(
            f"{truncate_or_pad_text(text, seq_len)}\n\nPlease summarize:")
    return prompts


def fit_token_ids(token_ids: list[int], target_seq_len: int,
                  fallback_token_id: int) -> list[int]:
    if not token_ids:
        token_ids = [fallback_token_id]
    if len(token_ids) >= target_seq_len:
        return token_ids[:target_seq_len]
    repeat = target_seq_len // len(token_ids) + 1
    return (token_ids * repeat)[:target_seq_len]


def prepare_token_prompts(dataset: Optional[Any], tokenizer: Any,
                          batch_size: int,
                          seq_len: int) -> list[TokensPrompt]:
    fallback_token_id = getattr(tokenizer, "eos_token_id", None)
    if fallback_token_id is None:
        fallback_token_id = getattr(tokenizer, "pad_token_id", None)
    if fallback_token_id is None:
        fallback_token_id = 0

    prompts: list[TokensPrompt] = []
    for _ in range(batch_size):
        if dataset is None:
            text = "This is a split benchmark prompt."
        else:
            sample = dataset[random.randint(0, len(dataset) - 1)]
            text = get_sample_text(sample)
        max_chars = max(1024, seq_len * 16)
        token_ids = tokenizer.encode(text[:max_chars], add_special_tokens=False)
        prompt_token_ids = fit_token_ids(token_ids, seq_len,
                                         fallback_token_id)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
    return prompts


def summarize_lengths(lengths: list[int]) -> dict[str, Optional[float]]:
    if not lengths:
        return {
            "actual_prompt_tokens_min": None,
            "actual_prompt_tokens_avg": None,
            "actual_prompt_tokens_max": None,
        }
    return {
        "actual_prompt_tokens_min": min(lengths),
        "actual_prompt_tokens_avg": sum(lengths) / len(lengths),
        "actual_prompt_tokens_max": max(lengths),
    }


def validate_prompt_token_lengths(lengths: list[int], seq_len: int) -> None:
    bad = [length for length in lengths if length != seq_len]
    if bad:
        raise ValueError(
            f"Expected every prompt to be {seq_len} tokens, but got "
            f"min={min(lengths)} max={max(lengths)} bad_sample={bad[:8]}")


def ceil_bucket(num_tokens: int, buckets: list[int]) -> int:
    for bucket in sorted(set(buckets)):
        if num_tokens <= bucket:
            return bucket
    return num_tokens


def micro_batch_size_from_buckets(buckets: list[int]) -> int:
    sorted_buckets = sorted(set(buckets))
    if not sorted_buckets:
        return 0
    if len(sorted_buckets) == 1:
        return sorted_buckets[0]
    return max(end - start for start, end in zip(
        sorted_buckets, sorted_buckets[1:]))


def dual_scheme(mode: str, batch_size: int, buckets: list[int],
                micro_batch_size: int,
                pad_threshold: int) -> tuple[bool, int, Optional[int],
                                             Optional[int], int, str]:
    padded = ceil_bucket(batch_size, buckets)
    first_candidates = [s for s in buckets if s < batch_size]
    if not first_candidates or batch_size > max(buckets):
        return False, padded, None, None, padded - batch_size, str(padded)

    first = max(first_candidates)
    second_raw = batch_size - first
    second_padded = ceil_bucket(second_raw, buckets)
    if mode == "DUAL_INPLACE":
        if padded == batch_size:
            return False, padded, None, None, padded - batch_size, str(padded)
        return True, first + second_raw, first, second_raw, 0, (
            f"{first}+{second_raw}")

    padding_saved = (
        padded - batch_size - (second_padded - second_raw))
    if padding_saved <= pad_threshold:
        return False, padded, None, None, padded - batch_size, str(padded)

    return True, first + second_padded, first, second_padded, (
        second_padded - second_raw), f"{first}+{second_padded}"


def expected_scheme(mode: str, batch_size: int, buckets: list[int],
                    micro_batch_size: int,
                    pad_threshold: int) -> dict[str, Any]:
    if mode in ("EAGER", "PADDING", "PADDING_MAX"):
        padded = ceil_bucket(batch_size, buckets)
        total = batch_size if mode == "EAGER" else padded
        return {
            "expected_total": total,
            "expected_first": None,
            "expected_second": None,
            "expected_split": False,
            "padding_tokens": total - batch_size,
            "scheme": str(total),
        }

    expected_split, total, first, second, padding_tokens, scheme = dual_scheme(
        mode, batch_size, buckets, micro_batch_size, pad_threshold)
    return {
        "expected_total": total,
        "expected_first": first,
        "expected_second": second,
        "expected_split": expected_split,
        "padding_tokens": padding_tokens,
        "scheme": scheme,
    }


def generate_anchor_batches(buckets: list[int], min_bs: int,
                            max_bs: int) -> list[int]:
    ratios = [0.125,0.25,0.375,0.5,0.625,0.75,0.875]
    batches: set[int] = set()
    prev = 0
    for bucket in sorted(set(buckets)):
        if bucket < min_bs:
            prev = bucket
            continue
        if bucket > max_bs:
            break
        lower = max(prev, min_bs)
        if lower >= bucket:
            # batches.add(bucket)
            prev = bucket
            continue
        gap = bucket - lower
        for ratio in ratios:
            batches.add(int(round(lower + gap * ratio)))
        # batches.add(bucket)
        prev = bucket
    return sorted(b for b in batches if min_bs <= b <= max_bs)


def generate_ratio_batches(buckets: list[int], min_bs: int, max_bs: int,
                           ratio_parts: int) -> list[int]:
    if ratio_parts <= 0:
        raise ValueError("--batch-ratio-parts must be positive")

    sorted_buckets = sorted(set(buckets))
    batches: set[int] = set()
    for start, end in zip(sorted_buckets, sorted_buckets[1:]):
        gap = end - start
        for part in range(ratio_parts + 1):
            batch = int(round(start + gap * part / ratio_parts))
            if min_bs <= batch <= max_bs:
                batches.add(batch)
    return sorted(batches)


def read_execute_rows(timing_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in timing_dir.glob("*_execute_model.csv"):
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def read_execute_file(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_execute_timing_file(mode_dir: Path, mode: str, batch_size: int,
                            seq_len: int) -> Path:
    return mode_dir / (
        f"{timing_mode_name(mode)}_batch{batch_size}_seq{seq_len}"
        "_execute_model.csv")


def matched_execute_times(rows: list[dict[str, str]], mode: str,
                          scheme: dict[str, Any]) -> list[float]:
    return [
        float(row.get("gpu_elapsed_ms") or row["elapsed_ms"])
        for row in matched_execute_rows(rows, mode, scheme)
    ]


def matched_execute_rows(rows: list[dict[str, str]], mode: str,
                         scheme: dict[str, Any]) -> list[dict[str, str]]:
    matched_rows: list[dict[str, str]] = []
    expected_replay_mode = timing_mode_name(mode)
    for row in rows:
        if row.get("replay_mode") != expected_replay_mode:
            continue
        if row.get("uniform_decode", "True") != "True":
            continue
        if row.get("split_for_cudagraph") != str(scheme["expected_split"]):
            continue
        if int(row["num_input_tokens"]) != scheme["expected_total"]:
            continue

        if not scheme["expected_split"]:
            matched_rows.append(row)
            continue

        if int(row.get("first_num_tokens") or 0) != scheme["expected_first"]:
            continue
        if int(row.get("second_num_tokens") or 0) != scheme["expected_second"]:
            continue
        matched_rows.append(row)
    return matched_rows


def write_seq_execute_rows(rows: list[dict[str, str]], output_dir: Path,
                           mode: str, seq_len: int,
                           scheme_label: str) -> Optional[Path]:
    if not rows:
        return None

    seq_dir = output_dir / "raw_timing_by_seq" / mode / f"seq{seq_len}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    path = seq_dir / f"{mode}_seq{seq_len}_{scheme_label}_execute_model.csv"

    fieldnames = [
        "test_seq_len", "sample_index", "replay_mode", "top_runtime_mode",
        "first_runtime_mode", "second_runtime_mode", "input_batch_size",
        "seq_len", "target_total", "target_first", "target_second",
        "uniform_decode", "split_for_cudagraph", "num_input_tokens",
        "first_num_tokens", "second_num_tokens", "gpu_elapsed_ms",
        "elapsed_ms", "phase_gpu_elapsed_ms", "phase_elapsed_ms",
        "head_overhead_ms"
    ]
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            out_row = {key: row.get(key, "") for key in fieldnames}
            out_row["test_seq_len"] = seq_len
            writer.writerow(out_row)
    return path


def summarize_times(times: list[float]) -> dict[str, Optional[float]]:
    if not times:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p99_ms": None,
        }
    sorted_times = sorted(times)
    p90_idx = round((len(sorted_times) - 1) * 0.9)
    return {
        "count": len(times),
        "mean_ms": sum(times) / len(times),
        "median_ms": statistics.median(times),
        "p90_ms": sorted_times[p90_idx],
        "p99_ms": sorted_times[round((len(sorted_times) - 1) * 0.99)],
    }


def prefixed_summary(prefix: str,
                     times: list[float]) -> dict[str, Optional[float]]:
    summary = summarize_times(times)
    return {
        f"{prefix}_count": summary["count"],
        f"{prefix}_mean_ms": summary["mean_ms"],
        f"{prefix}_p50_ms": summary["median_ms"],
        f"{prefix}_p90_ms": summary["p90_ms"],
        f"{prefix}_p99_ms": summary["p99_ms"],
    }


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "mode", "batch_size", "seq_len", "scheme", "expected_total",
        "expected_first", "expected_second", "expected_split",
        "padding_tokens",
        "padding_ratio", "capture_sizes", "micro_batch_size", "count",
        "mean_ms", "median_ms", "p90_ms",
        "p99_ms", "gpu_mean_ms", "gpu_p50_ms", "gpu_p90_ms", "gpu_p99_ms",
        "phase_count", "phase_mean_ms", "phase_p50_ms", "phase_p90_ms",
        "phase_p99_ms", "head_overhead_count", "head_overhead_mean_ms",
        "head_overhead_p50_ms", "head_overhead_p90_ms",
        "head_overhead_p99_ms",
        "seq_timing_file", "generate_time_s", "success", "error"
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})


def load_existing_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            return []
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception as exc:
            print(f"Failed to load existing summary {csv_path}: {exc}")
            return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"Failed to load existing summary {path}: {exc}")
        csv_path = path.with_suffix(".csv")
        if not csv_path.exists():
            return []
        try:
            with csv_path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception as csv_exc:
            print(f"Failed to load existing summary {csv_path}: {csv_exc}")
            return []


def drop_modes_from_summary(results: list[dict[str, Any]],
                            modes: set[str]) -> list[dict[str, Any]]:
    return [row for row in results if str(row.get("mode", "")).upper()
            not in modes]


def merge_summary_results(existing: list[dict[str, Any]],
                          current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in existing:
        try:
            key = (str(row["mode"]), int(row["batch_size"]),
                   int(row["seq_len"]))
        except (KeyError, TypeError, ValueError):
            continue
        merged[key] = row
    for row in current:
        key = (str(row["mode"]), int(row["batch_size"]), int(row["seq_len"]))
        merged[key] = row
    return [
        merged[key] for key in sorted(merged,
                                      key=lambda x: (x[0], x[2], x[1]))
    ]


def plot_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    valid = [r for r in results if r.get("median_ms") is not None]
    if not valid:
        return

    seq_lens = sorted({r["seq_len"] for r in valid})
    for seq_len in seq_lens:
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode in REPLAY_MODES:
            rows = sorted(
                [r for r in valid if r["mode"] == mode
                 and r["seq_len"] == seq_len],
                key=lambda r: r["batch_size"],
            )
            if not rows:
                continue
            ax.plot([r["padding_tokens"] for r in rows],
                    [r["median_ms"] for r in rows],
                    marker="o",
                    label=mode)
            for r in rows:
                ax.annotate(str(r["batch_size"]),
                            (r["padding_tokens"], r["median_ms"]),
                            fontsize=8)

        ax.set_title(f"execute_model median time vs padding, seq_len={seq_len}")
        ax.set_xlabel("padding tokens for selected scheme")
        ax.set_ylabel("execute_model median time (ms)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"padding_vs_time_seq{seq_len}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in REPLAY_MODES:
        rows = sorted([r for r in valid if r["mode"] == mode],
                      key=lambda r: (r["batch_size"], r["seq_len"]))
        if not rows:
            continue
        ax.scatter([r["batch_size"] for r in rows],
                   [r["median_ms"] for r in rows],
                   label=mode,
                   alpha=0.8)
    ax.set_title("execute_model median time vs batch size")
    ax.set_xlabel("batch size")
    ax.set_ylabel("execute_model median time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "batch_vs_time.png", dpi=160)
    plt.close(fig)

    for metric, ylabel, filename in (
        ("phase_mean_ms", "matched replay/eager phase mean time (ms)",
         "batch_vs_phase_time.png"),
        ("head_overhead_mean_ms", "execute total - matched phase mean time (ms)",
         "batch_vs_head_overhead.png"),
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        for mode in REPLAY_MODES:
            rows = sorted([
                r for r in valid
                if r["mode"] == mode and r.get(metric) is not None
            ], key=lambda r: (r["batch_size"], r["seq_len"]))
            if not rows:
                continue
            ax.plot([r["batch_size"] for r in rows],
                    [r[metric] for r in rows],
                    marker="o",
                    label=mode)
        ax.set_title(ylabel)
        ax.set_xlabel("batch size")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)


def load_dataset_or_none(dataset_path: str) -> Optional[Any]:
    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys"):
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            print(f"Using dataset split: {split_name}")
        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        print("Using synthetic prompts.")
        return None


def run_mode(mode: str, base_args: dict[str, Any], test_configs: list[tuple[int,
                                                                             int]],
             dataset: Optional[Any], buckets: list[int], output_dir: Path,
             micro_batch_size: int, pad_threshold: int, max_tokens: int,
             repeat: int, input_mode: str) -> list[dict[str, Any]]:
    print(f"\n{'=' * 90}\nRunning mode: {mode}\n{'=' * 90}")
    mode_dir = output_dir / "raw_timing" / mode
    seq_mode_dir = output_dir / "raw_timing_by_seq" / mode
    for stale_dir in (mode_dir, seq_mode_dir):
        if stale_dir.exists():
            print(f"Removing stale timing dir: {stale_dir}")
            shutil.rmtree(stale_dir)
    mode_dir.mkdir(parents=True, exist_ok=True)
    timing_context_file = mode_dir / "timing_context.json"

    os.environ["VLLM_EXECUTE_MODEL_TIMING"] = "1"
    os.environ["VLLM_TIMING_OUTPUT_DIR"] = str(mode_dir)
    os.environ["VLLM_EXECUTE_MODEL_TIMING_CONTEXT_FILE"] = str(
        timing_context_file)

    args = dict(base_args)
    compilation_config = dict(args["compilation_config"])
    compilation_config["cudagraph_capture_sizes"] = buckets
    if mode == "EAGER":
        compilation_config["cudagraph_mode"] = "NONE"
        compilation_config.pop("replay_mode", None)
    else:
        compilation_config["replay_mode"] = replay_mode_name(mode)
    compilation_config["cudagraph_split_pad_threshold"] = pad_threshold
    args["compilation_config"] = compilation_config

    llm = LLM(**args)
    tokenizer = llm.get_tokenizer() if input_mode == "tokens" else None
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     temperature=0.0,
                                     ignore_eos=True)
    results: list[dict[str, Any]] = []

    for idx, (batch_size, seq_len) in enumerate(test_configs, start=1):
        scheme = expected_scheme(mode, batch_size, buckets, micro_batch_size,
                                 pad_threshold)
        execute_timing_file = get_execute_timing_file(mode_dir, mode,
                                                      batch_size, seq_len)
        with timing_context_file.open("w", encoding="utf-8") as f:
            json.dump({
                "batch_size": batch_size,
                "seq_len": seq_len,
                "scheme": scheme["scheme"],
                "target_total": scheme["expected_total"],
                "target_first": scheme["expected_first"],
                "target_second": scheme["expected_second"],
                "target_split": scheme["expected_split"],
            }, f)
        before_count = len(read_execute_file(execute_timing_file))
        print(f"[{mode}] {idx}/{len(test_configs)} batch={batch_size} "
              f"seq={seq_len} scheme={scheme['scheme']} "
              f"padding={scheme['padding_tokens']}")

        if input_mode == "tokens":
            assert tokenizer is not None
            prompts = prepare_token_prompts(dataset, tokenizer, batch_size,
                                            seq_len)
            prepared_lengths = [
                len(prompt["prompt_token_ids"]) for prompt in prompts
            ]
            validate_prompt_token_lengths(prepared_lengths, seq_len)
        else:
            prompts = prepare_prompts(dataset, batch_size, seq_len)
            prepared_lengths = []
        error = None
        generate_time_s = 0.0
        actual_lengths: list[int] = []
        try:
            start = time.perf_counter()
            for _ in range(repeat):
                outputs = llm.generate(prompts, sampling_params)
                actual_lengths = [len(o.prompt_token_ids) for o in outputs]
                if input_mode == "tokens":
                    validate_prompt_token_lengths(actual_lengths, seq_len)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            generate_time_s = time.perf_counter() - start
        except Exception as exc:
            error = str(exc)
            print(f"Failed: {error}")

        rows = read_execute_file(execute_timing_file)[before_count:]
        matched_rows = matched_execute_rows(rows, mode, scheme)
        seq_timing_file = write_seq_execute_rows(
            matched_rows, output_dir, mode, seq_len, scheme["scheme"])
        times = [
            float(row.get("gpu_elapsed_ms") or row["elapsed_ms"])
            for row in matched_rows
        ]
        gpu_times = [
            float(row["gpu_elapsed_ms"]) for row in matched_rows
            if row.get("gpu_elapsed_ms")
        ]
        phase_times = [
            float(row["phase_gpu_elapsed_ms"]) for row in matched_rows
            if row.get("phase_gpu_elapsed_ms")
        ]
        head_overhead_times = [
            float(row["head_overhead_ms"]) for row in matched_rows
            if row.get("head_overhead_ms")
        ]
        summary = summarize_times(times)
        gpu_summary = prefixed_summary("gpu", gpu_times)
        phase_summary = prefixed_summary("phase", phase_times)
        head_overhead_summary = prefixed_summary("head_overhead",
                                                 head_overhead_times)
        result = {
            "mode": mode,
            "batch_size": batch_size,
            "seq_len": seq_len,
            **scheme,
            "padding_ratio": scheme["padding_tokens"] /
            max(1, scheme["expected_total"]),
            "capture_sizes": ",".join(str(x) for x in buckets),
            "micro_batch_size": micro_batch_size,
            "input_mode": input_mode,
            "verified_prompt_token_len": error is None
            and input_mode == "tokens",
            **summarize_lengths(actual_lengths or prepared_lengths),
            **summary,
            **gpu_summary,
            **phase_summary,
            **head_overhead_summary,
            "seq_timing_file": str(seq_timing_file)
            if seq_timing_file is not None else None,
            "generate_time_s": generate_time_s,
            "success": error is None,
            "error": error,
        }
        results.append(result)
        print(f"  matched_execute_rows={summary['count']} "
              f"median_ms={summary['median_ms']}")
        if actual_lengths:
            length_summary = summarize_lengths(actual_lengths)
            print("  actual_prompt_tokens="
                  f"{length_summary['actual_prompt_tokens_min']}/"
                  f"{length_summary['actual_prompt_tokens_avg']:.1f}/"
                  f"{length_summary['actual_prompt_tokens_max']}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return results


def main(args: dict[str, Any]) -> None:
    dataset_path = args.pop("dataset_path")
    modes = [m.strip().upper() for m in args.pop("modes").split(",")
             if m.strip()]
    capture_size_preset = args.pop("capture_size_preset")
    batch_sizes_arg = args.pop("batch_sizes")
    experiment1_batches = args.pop("experiment1_batches")
    padding_ratio_batches = args.pop("padding_ratio_batches")
    ratio_batches = args.pop("ratio_batches")
    batch_ratio_parts = args.pop("batch_ratio_parts")
    min_batch_size = args.pop("min_batch_size")
    max_batch_size = args.pop("max_batch_size")
    seq_lens_arg = args.pop("seq_lens")
    random_seq_len_range = args.pop("random_seq_len_range")
    seq_lens = parse_int_list(seq_lens_arg) if seq_lens_arg else []
    max_kv_tokens = args.pop("max_kv_tokens")
    max_tokens = args.pop("max_tokens")
    input_mode = args.pop("input_mode")
    repeat = args.pop("repeat")
    seed = args.pop("seed_r")
    output_dir = Path(args.pop("output_dir"))
    micro_batch_size = args.pop("micro_batch_size")
    pad_threshold = args.pop("cudagraph_split_pad_threshold")

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset_or_none(dataset_path)

    base_buckets = sorted(
        set(args["compilation_config"]["cudagraph_capture_sizes"]))
    args["compilation_config"]["cudagraph_split_pad_threshold"] = pad_threshold
    if batch_sizes_arg:
        batch_group = "custom"
        batch_sizes = parse_int_list(batch_sizes_arg)
    elif ratio_batches:
        batch_group = f"1/{batch_ratio_parts}ratio"
        batch_sizes = generate_ratio_batches(base_buckets, min_batch_size,
                                             max_batch_size,
                                             batch_ratio_parts)
    elif padding_ratio_batches:
        batch_group = "padding-ratio"
        batch_sizes = PADDING_RATIO_512_BATCH_SIZES
    elif experiment1_batches:
        batch_group = "experiment1"
        batch_sizes = EXPERIMENT1_BATCH_SIZES
    else:
        batch_group = "anchor"
        batch_sizes = generate_anchor_batches(base_buckets, min_batch_size,
                                              max_batch_size)
    # if args.get("max_num_seqs") is None:
    print(f"Max num seqs: {args.get('max_num_seqs')}")

    test_configs = []
    if random_seq_len_range:
        seq_min, seq_max = parse_int_range(random_seq_len_range)
        seq_by_batch = generate_stratified_seq_lens(batch_sizes, seq_min,
                                                    seq_max, seed)
        seq_lens = [seq_by_batch[batch_size] for batch_size in batch_sizes]
        for batch_size in batch_sizes:
            seq_len = seq_by_batch[batch_size]
            if batch_size * seq_len > max_kv_tokens:
                print(f"Skipping batch={batch_size}, seq={seq_len}: "
                      f"{batch_size * seq_len} > max_kv_tokens")
                continue
            test_configs.append((batch_size, seq_len))
    else:
        for seq_len in seq_lens:
            for batch_size in batch_sizes:
                if batch_size * seq_len > max_kv_tokens:
                    print(f"Skipping batch={batch_size}, seq={seq_len}: "
                          f"{batch_size * seq_len} > max_kv_tokens")
                    continue
                test_configs.append((batch_size, seq_len))

    print(f"Modes       : {modes}")
    print(f"Preset      : {capture_size_preset}")
    print(f"Batch group : {batch_group}")
    print(f"Batch sizes : {batch_sizes}")
    print(f"Seq lens    : {seq_lens}")
    print(f"Input mode  : {input_mode}")
    print(f"Pad thresh  : {pad_threshold}")
    print(f"Test points : {len(test_configs)}")
    print(f"Output dir  : {output_dir}")

    all_results: list[dict[str, Any]] = []
    for mode in modes:
        if mode not in REPLAY_MODES:
            raise ValueError(f"Unsupported mode {mode}; expected {REPLAY_MODES}")
        buckets = capture_sizes_for_mode(mode, base_buckets,
                                         capture_size_preset)
        mode_micro_batch_size = micro_batch_size
        if mode_micro_batch_size is None:
            mode_micro_batch_size = micro_batch_size_from_buckets(buckets)
        print(f"[{mode}] capture_sizes={buckets}")
        print(f"[{mode}] micro_batch_size={mode_micro_batch_size}")
        all_results.extend(
            run_mode(mode, args, test_configs, dataset, buckets, output_dir,
                     mode_micro_batch_size, pad_threshold, max_tokens,
                     repeat, input_mode))

    summary_csv = output_dir / "split_mode_execute_summary.csv"
    summary_json = output_dir / "split_mode_execute_summary.json"
    existing_results = drop_modes_from_summary(load_existing_summary(
        summary_json), set(modes))
    merged_results = merge_summary_results(existing_results, all_results)
    write_summary_csv(merged_results, summary_csv)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=2)
    plot_results(merged_results, output_dir)

    print(f"\nSaved summary: {summary_csv}")
    print(f"Saved plots  : {output_dir}")


if __name__ == "__main__":
    parser = create_parser()
    main(vars(parser.parse_args()))
