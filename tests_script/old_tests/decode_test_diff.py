# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import time
from typing import Optional

from datasets import load_from_disk
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

# 用做差的方法测decode时间 不稳定

def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    
    # cudagraph_sizes = [1, 2, 4, 8, 16, 32, 64] + [i * 128 for i in range(1, 6)]
    cudagraph_sizes = [1, 2, 4, 8, 16, 32, 64,128] + [i * 256 for i in range(1, 4)] + [896]
    
    
    compilation_config = {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
        # "replay_mode": "DUAL_MIXED",
        # "replay_mode": "DUAL_SERIAL",
        # "replay_mode": "DUAL_PARALLEL", 
        # "replay_mode": "PADDING",
        "replay_mode": "DUAL_INPLACE",
    }
    parser.set_defaults(compilation_config=compilation_config)
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=16384)

    # 关键设置：禁用分块预填充和连续批处理
    parser.set_defaults(enable_chunked_prefill=False)
    parser.set_defaults(disable_log_stats=True)
    
    # Disable continuous batching to simplify testing logic
    # 禁用前缀匹配
    parser.set_defaults(enable_prefix_caching=False)
    
    # Test parameters
    test_group = parser.add_argument_group("Test parameters")
    test_group.add_argument("--dataset-path", type=str, 
                           default="/home/csh/data/projects/datasets/LongBench-v2",
                           help="Path to LongBench-v2 dataset")
    test_group.add_argument("--min-batch-size", type=int, default=256,
                           help="Minimum batch size to start observing")
    test_group.add_argument("--max-batch-size", type=int, default=896,
                           help="Maximum batch size")
    test_group.add_argument("--seq-lens", type=str, default="1024",
                           help="Comma-separated sequence lengths to test")
    test_group.add_argument("--max-kv-tokens", type=int, default=2097152,
                           help="Skip tests where batch*seq exceeds this")
    test_group.add_argument("--warmup-rounds", type=int, default=0,
                           help="Number of warmup rounds before measurement")
    test_group.add_argument("--baseline-tokens", type=int, default=32,
                           help="Baseline decode tokens ")
    test_group.add_argument("--target-tokens", type=int, default=132,
                           help="Target decode tokens to measure")
    test_group.add_argument("--seed_r", type=int, default=42,
                           help="Random seed")

    return parser


def truncate_or_pad_text(text: str, target_tokens: int, 
                         avg_chars_per_token: float = 4.0) -> str:
    """截断或重复文本以达到目标 token 数"""
    target_chars = int(target_tokens * avg_chars_per_token)
    
    if len(text) >= target_chars:
        return text[:target_chars]
    else:
        repeated = text * (target_chars // len(text) + 1)
        return repeated[:target_chars]


def prepare_prompts_from_dataset(dataset, batch_size: int, 
                                  target_seq_len: int) -> list[str]:
    """从数据集中准备指定数量和长度的 prompts"""
    prompts = []
    dataset_size = len(dataset)
    
    for i in range(batch_size):
        idx = random.randint(0, dataset_size - 1)
        sample = dataset[idx]
        
        if 'context' in sample:
            text = sample['context']
        elif 'input' in sample:
            text = sample['input']
        elif 'question' in sample:
            text = sample['question']
        else:
            text = str(list(sample.values())[0])
        
        adjusted_text = truncate_or_pad_text(text, target_seq_len)
        prompt = f"{adjusted_text}\n\nPlease summarize:"
        prompts.append(prompt)
    
    return prompts


def measure_decode_time(llm: LLM, prompts: list[str], 
                       warmup_rounds: int,
                       baseline_tokens: int,
                       target_tokens: int,
                       test_id: int, 
                       batch_size: int,
                       seq_len: int) -> dict:
    """
    使用差分法测量纯 decode 时间
    
    Args:
        llm: LLM 实例
        prompts: 输入 prompts
        warmup_rounds: CUDA Graph 预热轮数
        baseline_tokens: 基准 decode token 数 (为了消除连续批处理尾部的动态batch)
        target_tokens: 目标 decode token 数
        test_id: 测试编号
        batch_size: 批次大小
        seq_len: 序列长度
    
    Returns:
        包含测量结果的字典
    """
    
    print(f"\n{'='*70}")
    print(f"Test Case {test_id} | Batch: {batch_size} | SeqLen: {seq_len}")
    print(f"{'='*70}")

    import torch
    
    try:
       
        # ==========================================
        # 步骤 0: CUDA Graph 预热
        # ==========================================
        if warmup_rounds > 0:
            print(f"  [0/3] Warming up CUDA graphs ({warmup_rounds} rounds)...")
            warmup_params = SamplingParams(
                max_tokens=baseline_tokens, 
                temperature=0.0, 
                ignore_eos=True
            )
            
            for i in range(warmup_rounds):
                _ = llm.generate(prompts, warmup_params)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            print(f"       Warmup completed ✓")
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ==========================================
        # 步骤 1: 测量 Prefill + Baseline Tokens
        # ==========================================
        print(f"  [1/3] Running Baseline (Prefill + {baseline_tokens} decode)...")
        baseline_params = SamplingParams(
            max_tokens=baseline_tokens, 
            temperature=0.0, 
            ignore_eos=True
        )
        
        start_t1 = time.perf_counter()
        outputs_baseline = llm.generate(prompts, baseline_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_t1 = time.perf_counter()
        
        time_baseline = end_t1 - start_t1
        baseline_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs_baseline)
        
        print(f"       Time: {time_baseline:.4f}s | Tokens: {baseline_output_tokens}")
        
        # 清理状态,准备下一次测量
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # ==========================================
        # 步骤 2: 测量 Prefill + Target Tokens
        # ==========================================
        print(f"  [2/3] Running Target (Prefill + {target_tokens} decode)...")
        target_params = SamplingParams(
            max_tokens=target_tokens, 
            temperature=0.0, 
            ignore_eos=True
        )
        
        start_tn = time.perf_counter()
        outputs_target = llm.generate(prompts, target_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_tn = time.perf_counter()
        
        time_target = end_tn - start_tn
        target_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs_target)
        
        print(f"       Time: {time_target:.4f}s | Tokens: {target_output_tokens}")

        # ==========================================
        # 步骤 3: 计算差分结果
        # ==========================================
        print(f"  [3/3] Calculating differential metrics...")
        
        # 纯 decode 时间和 token 数
        pure_decode_time = time_target - time_baseline
        decode_steps = target_tokens - baseline_tokens
        decode_tokens_generated = target_output_tokens - baseline_output_tokens
        
        # 合理性检查
        if pure_decode_time < 0:
            print(f"       ⚠️  Warning: Negative decode time detected, setting to minimal value")
            pure_decode_time = 0.0001
        
        # if time_baseline > time_target * 0.5:
        #     baseline_ratio = time_baseline / time_target * 100
        #     print(f"       ⚠️  Warning: Baseline占比过高 ({baseline_ratio:.1f}%)")
        
        # 计算性能指标
        tpot = pure_decode_time / decode_steps  # Time Per Output Token (batch-level)
        throughput = (batch_size * decode_steps) / pure_decode_time  # tokens/s
        
        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "seq_len": seq_len,
            
            # 时间测量
            "baseline_time": time_baseline,
            "target_time": time_target,
            "pure_decode_time": pure_decode_time,
            
            # Token 统计
            "baseline_tokens": baseline_tokens,
            "target_tokens": target_tokens,
            "decode_steps": decode_steps,
            "actual_tokens_generated": decode_tokens_generated,
            
            # 性能指标
            "ttft_approx": time_baseline,  # Time To First Token (近似)
            "tpot": tpot,  # Time Per Output Token (ms)
            "throughput": throughput,  # tokens/s
            
            "success": True,
            "error": None
        }
        
        print(f"\n  ✅ Results:")
        print(f"     TTFT (approx)        : {time_baseline:.4f}s")
        print(f"     Pure Decode Time     : {pure_decode_time:.4f}s ({decode_steps} steps)")
        print(f"     TPOT (batch-level)   : {tpot*1000:.2f} ms/token")
        print(f"     Decode Throughput    : {throughput:.2f} tokens/s")
        print(f"     Token Verification   : Expected {decode_steps}, Got {decode_tokens_generated}")
        
    except Exception as e:
        print(f"\n  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        result = {
            "success": False, 
            "error": str(e), 
            "batch_size": batch_size,
            "seq_len": seq_len
        }
    
    return result


def get_matched_bucket(batch_size: int, buckets: list[int]) -> int:
    """Return the smallest bucket >= batch_size (padding up)."""
    for bucket in sorted(buckets):
        if batch_size <= bucket:
            return bucket
    return batch_size


def generate_anchor_batches(buckets: list[int], min_bs: int,
                            max_bs: int) -> list[int]:
    """Generate anchor batch sizes based on inter-bucket gaps.

    For each bucket B, define lower L as the previous bucket (or min_bs if the
    previous bucket is smaller). Then sample:
      batch = L + (B - L) * ratio
    This expresses "padding amount as % of the bucket gap".
    """
    sorted_buckets = sorted(set(buckets))
    # ratio is "how far we are from lower to bucket".
    # e.g. between 128 and 256, ratio=0.5 -> 128 + (256-128)*0.5 = 192 (pad=64).
    ratios = [0.75, 0.50, 0.25]

    test_batches: set[int] = set()
    prev: Optional[int] = None
    for bucket in sorted_buckets:
        if bucket <= min_bs or bucket > max_bs:
            prev = bucket
            continue

        lower = min_bs if prev is None else max(min_bs, prev)
        if lower >= bucket:
            test_batches.add(bucket)
            prev = bucket
            continue

        gap = bucket - lower
        for r in ratios:
            batch = int(round(lower + gap * r))
            batch = min(max(batch, min_bs), max_bs)
            if batch <= bucket:
                test_batches.add(batch)
        prev = bucket

    return sorted(test_batches)


def main(args: dict):
    # Pop test parameters
    dataset_path = args.pop("dataset_path")
    min_batch_size = args.pop("min_batch_size")
    max_batch_size = args.pop("max_batch_size")
    seq_lens_str = args.pop("seq_lens")
    max_kv_tokens = args.pop("max_kv_tokens")
    warmup_rounds = args.pop("warmup_rounds")
    baseline_tokens = args.pop("baseline_tokens")
    target_tokens = args.pop("target_tokens")
    seed = args.pop("seed_r")
    
    random.seed(seed)
    seq_lens = [int(s.strip()) for s in seq_lens_str.split(",") if s.strip()]
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, 'keys'):
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
            print(f"Using split: {split_name}")
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        dataset = None
    
    # Create LLM
    print("\nInitializing LLM...")
    llm = LLM(**args)
    
    # Extract CUDA Graph buckets (padding targets)
    cudagraph_sizes = args.get("compilation_config",
                              {}).get("cudagraph_capture_sizes", [])
    if not cudagraph_sizes:
        # cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)] + [992]
        cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 128 for i in range(1, 8)]
    # Generate anchor batch sizes around each bucket
    batch_sizes = generate_anchor_batches(cudagraph_sizes, min_batch_size,
                                         max_batch_size)

    # Build test matrix (batch, seq_len), with KV cache guardrail
    test_configs: list[tuple[int, int]] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            if batch_size * seq_len > max_kv_tokens:
                print(f"Skipping (Batch: {batch_size}, Seq: {seq_len}) -> "
                      f"{batch_size * seq_len} tokens exceeds max_kv_tokens="
                      f"{max_kv_tokens}.")
                continue
            test_configs.append((batch_size, seq_len))
    
    print(f"\n{'='*80}")
    print("Test Configuration: Anchor Matrix")
    print(f"{'='*80}")
    print(f"Total test points    : {len(test_configs)}")
    print(f"Test batch anchors   : {batch_sizes}")
    print(f"Sequence lengths     : {seq_lens}")
    print(f"Baseline tokens      : {baseline_tokens}")
    print(f"Target tokens        : {target_tokens}")
    print(f"Decode steps measured: {target_tokens - baseline_tokens}")
    print(f"Warmup rounds        : {warmup_rounds}")
    print(f"Method               : Differential (Time[{target_tokens}] - Time[{baseline_tokens}])")
    
    # Run tests
    results = []
    warmed_buckets: set[int] = set()
    for i, (batch_size, seq_len) in enumerate(test_configs):
        # 清理之前的 LLM 实例和 GPU 资源
        # del llm
        # print("\nCleaning up GPU resources...")
        # import torch
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     torch.cuda.synchronize()
        
        # import gc
        # gc.collect()
        
        # time.sleep(1)  # 等待资源释放

        print(f"\n[{i+1}/{len(test_configs)}] Testing Batch: {batch_size}, Seq: {seq_len} ...")
        
        # llm = LLM(**args)

        if dataset is not None:
            prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len)
        else:
            base_text = "This is a test prompt. " * (seq_len // 5)
            prompts = [f"{base_text}\n\nSummarize:" for _ in range(batch_size)]
        
        matched_bucket = get_matched_bucket(batch_size, cudagraph_sizes)
        warmup_for_this = warmup_rounds if matched_bucket not in warmed_buckets else 0

        result = measure_decode_time(
            llm=llm,
            prompts=prompts,
            warmup_rounds=warmup_for_this,
            baseline_tokens=baseline_tokens,
            target_tokens=target_tokens,
            test_id=i + 1,
            batch_size=batch_size,
            seq_len=seq_len
        )
        # matched_bucket = get_matched_bucket(batch_size, cudagraph_sizes)
        result["matched_bucket"] = matched_bucket
        result["padding_batch"] = matched_bucket - batch_size
        result["padding_waste_pct"] = (matched_bucket - batch_size) / matched_bucket * 100
        results.append(result)

        warmed_buckets.add(matched_bucket)
        # Best-effort cache cleanup to reduce fragmentation without resetting graphs.
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Summary
    print(f"\n{'='*100}")
    print("Test Summary")
    print(f"{'='*100}")
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"Total tests : {len(results)}")
    print(f"Successful  : {len(successful)}")
    print(f"Failed      : {len(failed)}")
    
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - Batch {r['batch_size']}: {r.get('error', 'Unknown error')}")
    
    # Detailed results table
    print(f"\n{'='*120}")
    print("Detailed Results")
    print(f"{'='*120}")
    print(f"{'Batch':<8} {'Bucket':<8} {'Pad':<6} {'Waste%':<8} {'SeqLen':<8} {'TTFT(s)':<10} {'Decode(s)':<10} {'TPOT(ms)':<10} {'Throughput':<12} {'Status':<8}")
    print("-" * 120)
    
    for r in results:
        status = "✅" if r.get('success', False) else "❌"
        
        if r.get('success', False):
            ttft = r.get('ttft_approx', 0.0)
            pure_time = r.get('pure_decode_time', 0.0)
            tpot = r.get('tpot', 0.0) * 1000  # ms
            throughput = r.get('throughput', 0.0)
            
            print(f"{r['batch_size']:<8} "
                  f"{r.get('matched_bucket', '-'):<8} "
                  f"{r.get('padding_batch', 0):<6} "
                  f"{r.get('padding_waste_pct', 0.0):<7.1f}% "
                  f"{r['seq_len']:<8} "
                  f"{ttft:<10.4f} "
                  f"{pure_time:<10.4f} "
                  f"{tpot:<10.2f} "
                  f"{throughput:<12.2f} "
                  f"{status:<8}")
        else:
            print(f"{r['batch_size']:<8} "
                  f"{r.get('matched_bucket', '-'):<8} "
                  f"{'-':<6} "
                  f"{'-':<8} "
                  f"{r.get('seq_len', '-'):<8} "
                  f"{'-':<10} {'-':<10} {'-':<10} {'-':<12} "
                  f"{status:<8}")
    
    # 保存结果到 JSON
    import json
    output_file = "decode_benchmark_results_inplace_4b_1024_256.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*120}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)