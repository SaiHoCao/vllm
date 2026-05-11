# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import time
from typing import Any, Optional

from datasets import load_from_disk

from vllm import EngineArgs, LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.utils import FlexibleArgumentParser

# 用来测试单个规模
# example 
# python tests_script/cudagraph_test.py --cudagraph-mode FULL_DECODE_ONLY --replay-mode PADDING --batch-size 100 --seq-len 100

DEFAULT_CUDAGRAPH_SIZES = [1, 2, 4, 8, 16, 32, 64] + [
    i * 128 for i in range(1, 5)]
# [1,2,4,8,16,32,64,128,192,256,384,512]

CUDA_GRAPH_MODES = ("NONE", "FULL_DECODE_ONLY", "FULL", "PIECEWISE", "FULL_AND_PIECEWISE")
REPLAY_MODES = ("PADDING", "DUAL_PARALLEL", "DUAL_INPLACE")
INPUT_MODES = ("text", "tokens")


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)

    parser.set_defaults(
        compilation_config={
            "level": "3",
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": DEFAULT_CUDAGRAPH_SIZES,
        })
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=16384)
    parser.set_defaults(enable_chunked_prefill=False)
    parser.set_defaults(enable_prefix_caching=False)

    test_group = parser.add_argument_group("Test parameters")
    test_group.add_argument(
        "--dataset-path",
        type=str,
        default="/home/csh/data/projects/datasets/LongBench-v2",
        help="Path to a dataset saved by datasets.save_to_disk().")
    test_group.add_argument("--replay-mode",
                            type=lambda value: value.upper(),
                            default="PADDING",
                            choices=REPLAY_MODES,
                            help="Replay mode to test when cudagraph is on.")
    test_group.add_argument("--cudagraph-mode",
                            type=lambda value: value.upper(),
                            default="NONE",
                            choices=CUDA_GRAPH_MODES,
                            help="CUDA graph mode. Default NONE is safest; "
                            "use FULL_DECODE_ONLY to test decode graphs.")
    test_group.add_argument(
        "--capture-sizes",
        type=str,
        default="",
        help="Comma-separated cudagraph capture sizes. Defaults to the script "
        "capture list.")
    test_group.add_argument("--input-mode",
                            type=lambda value: value.lower(),
                            default="text",
                            choices=INPUT_MODES,
                            help="text keeps normal vLLM tokenization; tokens "
                            "passes exact prompt_token_ids of length seq_len.")
    test_group.add_argument("--batch-size",
                            type=int,
                            default=256,
                            help="Single batch size to test.")
    test_group.add_argument("--seq-len",
                            type=int,
                            default=1024,
                            help="Single prompt length in tokens.")
    test_group.add_argument(
        "--batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes. Overrides --batch-size when set.")
    test_group.add_argument(
        "--seq-lens",
        type=str,
        default="",
        help="Comma-separated sequence lengths. Overrides --seq-len when set.")
    test_group.add_argument(
        "--sweep-batch-range",
        action="store_true",
        help="Sweep even batch sizes in [--min-batch-size, --max-batch-size].")
    test_group.add_argument("--min-batch-size", type=int, default=100)
    test_group.add_argument("--max-batch-size", type=int, default=512)
    test_group.add_argument(
        "--random-seq-len",
        action="store_true",
        help="Randomize seq_len in [--min-seq-len, --max-seq-len].")
    test_group.add_argument("--min-seq-len", type=int, default=1024)
    test_group.add_argument("--max-seq-len", type=int, default=2048)
    test_group.add_argument(
        "--no-align-scheduler-limits",
        action="store_false",
        dest="align_scheduler_limits",
        help="Do not auto-set max_num_seqs from the requested batch shape.")
    test_group.add_argument(
        "--align-max-num-batched-tokens",
        action="store_true",
        help="Also set max_num_batched_tokens to at least batch_size * "
        "seq_len. This can make profile_run very large and is off by "
        "default.")
    test_group.add_argument("--max-tokens",
                            type=int,
                            default=128,
                            help="Maximum generation length per request.")
    test_group.add_argument("--seed-r",
                            "--seed_r",
                            dest="seed_r",
                            type=int,
                            default=42,
                            help="Random seed.")
    return parser


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ceil_bucket(num_tokens: int, buckets: list[int]) -> int:
    for bucket in sorted(set(buckets)):
        if num_tokens <= bucket:
            return bucket
    return num_tokens


def expected_scheme(mode: str, batch_size: int,
                    buckets: list[int]) -> tuple[str, int]:
    if mode is None:
        return "none", 0

    if mode == "PADDING":
        padded = ceil_bucket(batch_size, buckets)
        return str(padded), padded - batch_size

    first_candidates = [size for size in buckets if size < batch_size]
    if not first_candidates:
        padded = ceil_bucket(batch_size, buckets)
        return str(padded), padded - batch_size

    first = max(first_candidates)
    second_raw = batch_size - first
    if mode == "DUAL_INPLACE":
        return f"{first}+{second_raw}", 0

    second = ceil_bucket(second_raw, buckets)
    return f"{first}+{second}", second - second_raw


def build_test_configs(args: dict[str, Any]) -> list[tuple[int, int]]:
    if args["batch_sizes"]:
        batch_sizes = parse_int_list(args["batch_sizes"])
    elif args["sweep_batch_range"]:
        batch_sizes = []
        if args["min_batch_size"] <= 1:
            batch_sizes.append(1)
        start = max(2, args["min_batch_size"])
        if start % 2:
            start += 1
        batch_sizes.extend(range(start, args["max_batch_size"] + 1, 2))
        if args["max_batch_size"] >= args["min_batch_size"]:
            batch_sizes.append(args["max_batch_size"])
        batch_sizes = sorted(set(batch_sizes))
    else:
        batch_sizes = [args["batch_size"]]

    if args["seq_lens"]:
        seq_lens = parse_int_list(args["seq_lens"])
    elif args["random_seq_len"]:
        seq_lens = []
    else:
        seq_lens = [args["seq_len"]]

    configs: list[tuple[int, int]] = []
    for batch_size in batch_sizes:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if args["random_seq_len"]:
            configs.append(
                (batch_size,
                 random.randint(args["min_seq_len"], args["max_seq_len"])))
        else:
            for seq_len in seq_lens:
                if seq_len <= 0:
                    raise ValueError("seq_len must be positive")
                configs.append((batch_size, seq_len))
    return configs


def load_dataset_or_none(dataset_path: str) -> Optional[Any]:
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, "keys"):
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            print(f"Using split: {split_name}")
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Dataset columns: {dataset.column_names}")
        return dataset
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        print("Using fallback synthetic token prompts.")
        return None


def get_sample_text(sample: dict[str, Any]) -> str:
    for key in ("context", "input", "question"):
        if key in sample:
            return str(sample[key])
    return str(list(sample.values())[0])


def truncate_or_repeat_text(text: str, target_tokens: int,
                            avg_chars_per_token: float = 4.0) -> str:
    target_chars = max(1, int(target_tokens * avg_chars_per_token))
    if len(text) >= target_chars:
        return text[:target_chars]
    repeated = text * (target_chars // max(1, len(text)) + 1)
    return repeated[:target_chars]


def prepare_text_prompts(dataset: Optional[Any], batch_size: int,
                         target_seq_len: int) -> list[str]:
    prompts: list[str] = []
    for _ in range(batch_size):
        if dataset is None:
            text = "This is a split benchmark prompt."
        else:
            sample = dataset[random.randint(0, len(dataset) - 1)]
            text = get_sample_text(sample)
        adjusted_text = truncate_or_repeat_text(text, target_seq_len)
        prompts.append(
            f"{adjusted_text}\n\nPlease summarize the above text briefly:")
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
                          target_seq_len: int) -> list[TokensPrompt]:
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
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        prompt_token_ids = fit_token_ids(token_ids, target_seq_len,
                                         fallback_token_id)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
    return prompts


def align_scheduler_limits(args: dict[str, Any],
                           test_configs: list[tuple[int, int]],
                           align_token_budget: bool) -> None:
    max_batch_size = max(batch_size for batch_size, _ in test_configs)

    if args.get("max_num_seqs") is None:
        args["max_num_seqs"] = max_batch_size

    if align_token_budget and args.get("max_num_batched_tokens") is None:
        max_seq_len = max(seq_len for _, seq_len in test_configs)
        args["max_num_batched_tokens"] = max(
            args.get("max_model_len") or 0,
            max_batch_size * max_seq_len,
        )


def run_test_case(llm: LLM, prompts: list[Any],
                  sampling_params: SamplingParams, test_id: int,
                  batch_size: int, target_seq_len: int) -> dict[str, Any]:
    print(f"\n{'=' * 70}")
    print(f"Test Case {test_id}")
    print(f"{'=' * 70}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target sequence length: {target_seq_len} tokens")
    print(f"  Number of prompts: {len(prompts)}")

    start_time = time.perf_counter()
    try:
        outputs = llm.generate(prompts, sampling_params)
        elapsed_time = time.perf_counter() - start_time
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        input_lengths = [len(o.prompt_token_ids) for o in outputs]

        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "actual_input_tokens": total_input_tokens,
            "min_input_len": min(input_lengths),
            "max_input_len": max(input_lengths),
            "output_tokens": total_output_tokens,
            "elapsed_time": elapsed_time,
            "throughput_input": total_input_tokens / elapsed_time,
            "throughput_output": total_output_tokens / elapsed_time,
            "success": True,
            "error": None,
        }

        print("  Success")
        print(f"  Actual input tokens: {total_input_tokens}")
        print(f"  Input length range: [{result['min_input_len']}, "
              f"{result['max_input_len']}]")
        print(f"  Output tokens: {total_output_tokens}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Input throughput: {result['throughput_input']:.2f} tokens/s")
        print(f"  Output throughput: {result['throughput_output']:.2f} tokens/s")

        print("\n  Sample outputs:")
        for i, output in enumerate(outputs[:10]):
            generated_text = output.outputs[0].text.strip().replace("\n", " ")
            print(f"    [{i}] {generated_text}")
    except Exception as exc:
        elapsed_time = time.perf_counter() - start_time
        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "actual_input_tokens": 0,
            "min_input_len": 0,
            "max_input_len": 0,
            "output_tokens": 0,
            "elapsed_time": elapsed_time,
            "throughput_input": 0,
            "throughput_output": 0,
            "success": False,
            "error": str(exc),
        }
        print(f"  Failed: {exc}")
        import traceback
        traceback.print_exc()

    return result


def print_test_plan(test_configs: list[tuple[int, int]], args: dict[str, Any],
                    max_tokens: int, input_mode: str) -> None:
    batch_sizes = sorted(set(batch_size for batch_size, _ in test_configs))
    seq_lens = sorted(set(seq_len for _, seq_len in test_configs))

    print(f"\n{'=' * 70}")
    print("Test Plan")
    print(f"{'=' * 70}")
    print(f"Number of tests: {len(test_configs)}")
    print(f"Batch sizes: {batch_sizes[:10]}...{batch_sizes[-5:]}"
          if len(batch_sizes) > 15 else f"Batch sizes: {batch_sizes}")
    print(f"Sequence lengths: {seq_lens[:10]}...{seq_lens[-5:]}"
          if len(seq_lens) > 15 else f"Sequence lengths: {seq_lens}")
    print(f"max_num_seqs: {args.get('max_num_seqs')}")
    print(f"max_num_batched_tokens: {args.get('max_num_batched_tokens')}")
    print(f"CUDA graph mode: "
          f"{args['compilation_config'].get('cudagraph_mode')}")
    print(f"Capture sizes: "
          f"{args['compilation_config']['cudagraph_capture_sizes']}")
    replay_mode = args["compilation_config"].get("replay_mode")
    print(f"Replay mode: {replay_mode}")
    print(f"Input mode: {input_mode}")
    print(f"Max generation tokens: {max_tokens}")

    print("\nTest configurations:")
    configs_to_show = (test_configs[:10] + test_configs[-5:]
                       if len(test_configs) > 15 else test_configs)
    for i, (batch_size, seq_len) in enumerate(configs_to_show[:10]):
        scheme, padding = expected_scheme(
            replay_mode, batch_size,
            args["compilation_config"]["cudagraph_capture_sizes"])
        print(f"  Test {i + 1}: batch_size={batch_size}, seq_len={seq_len}, "
              f"scheme={scheme}, padding={padding}")
    if len(test_configs) > 15:
        print(f"  ... ({len(test_configs) - 15} more tests) ...")
        for i, (batch_size, seq_len) in enumerate(test_configs[-5:]):
            print(f"  Test {len(test_configs) - 4 + i}: "
                  f"batch_size={batch_size}, seq_len={seq_len}")


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 70}")
    print("Test Summary")
    print(f"{'=' * 70}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        avg_input_throughput = (
            sum(r["throughput_input"] for r in successful) / len(successful))
        avg_output_throughput = (
            sum(r["throughput_output"] for r in successful) / len(successful))
        print(f"\nAverage input throughput: {avg_input_throughput:.2f} tokens/s")
        print(f"Average output throughput: {avg_output_throughput:.2f} tokens/s")

    if failed:
        print("\nFailed tests:")
        for result in failed:
            print(f"  Test {result['test_id']}: {result['error']}")

    print(f"\n{'=' * 70}")
    print("Detailed Results")
    print(f"{'=' * 70}")
    print(f"{'Test':<6} {'Batch':<8} {'SeqLen':<8} {'InToks':<10} "
          f"{'InRange':<15} {'OutToks':<10} {'Time':<8} {'Status':<8}")
    print("-" * 86)
    for result in results:
        status = "OK" if result["success"] else "FAIL"
        input_range = f"{result['min_input_len']}-{result['max_input_len']}"
        print(f"{result['test_id']:<6} {result['batch_size']:<8} "
              f"{result['target_seq_len']:<8} "
              f"{result['actual_input_tokens']:<10} {input_range:<15} "
              f"{result['output_tokens']:<10} "
              f"{result['elapsed_time']:<8.2f} {status:<8}")


def main(args: dict[str, Any]) -> None:
    dataset_path = args.pop("dataset_path")
    cudagraph_mode = args.pop("cudagraph_mode")
    replay_mode = args.pop("replay_mode")
    capture_sizes_arg = args.pop("capture_sizes")
    input_mode = args.pop("input_mode")
    max_tokens = args.pop("max_tokens")
    seed = args.pop("seed_r")
    should_align_scheduler_limits = args.pop("align_scheduler_limits")
    should_align_token_budget = args.pop("align_max_num_batched_tokens")

    random.seed(seed)

    test_config_args = {
        "batch_size": args.pop("batch_size"),
        "seq_len": args.pop("seq_len"),
        "batch_sizes": args.pop("batch_sizes"),
        "seq_lens": args.pop("seq_lens"),
        "sweep_batch_range": args.pop("sweep_batch_range"),
        "min_batch_size": args.pop("min_batch_size"),
        "max_batch_size": args.pop("max_batch_size"),
        "random_seq_len": args.pop("random_seq_len"),
        "min_seq_len": args.pop("min_seq_len"),
        "max_seq_len": args.pop("max_seq_len"),
    }
    test_configs = build_test_configs(test_config_args)

    capture_sizes = (parse_int_list(capture_sizes_arg)
                     if capture_sizes_arg else DEFAULT_CUDAGRAPH_SIZES)
    args["compilation_config"] = dict(args["compilation_config"])
    args["compilation_config"]["cudagraph_mode"] = cudagraph_mode
    args["compilation_config"]["cudagraph_capture_sizes"] = capture_sizes
    if cudagraph_mode == "NONE":
        args["compilation_config"].pop("replay_mode", None)
    else:
        args["compilation_config"]["replay_mode"] = replay_mode

    if should_align_scheduler_limits:
        align_scheduler_limits(args, test_configs, should_align_token_budget)

    dataset = load_dataset_or_none(dataset_path)
    print_test_plan(test_configs, args, max_tokens, input_mode)

    print("\nInitializing LLM...")
    llm = LLM(**args)
    tokenizer = llm.get_tokenizer() if input_mode == "tokens" else None

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    results = []
    for i, (batch_size, seq_len) in enumerate(test_configs):
        if input_mode == "tokens":
            assert tokenizer is not None
            prompts = prepare_token_prompts(dataset, tokenizer, batch_size,
                                            seq_len)
        else:
            prompts = prepare_text_prompts(dataset, batch_size, seq_len)
        result = run_test_case(
            llm=llm,
            prompts=prompts,
            sampling_params=sampling_params,
            test_id=i + 1,
            batch_size=batch_size,
            target_seq_len=seq_len,
        )
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    parser = create_parser()
    parsed_args: dict[str, Any] = vars(parser.parse_args())
    main(parsed_args)
