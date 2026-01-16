# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import time
from typing import Optional

from datasets import load_from_disk
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    
    cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)]
    
    compilation_config = {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
        "enable_cudagraph_split": True,
        "enable_dual_graph": True,
    }
    parser.set_defaults(compilation_config=compilation_config)
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=8192)

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
    test_group.add_argument("--min-batch-size", type=int, default=50,
                           help="Minimum batch size")
    test_group.add_argument("--max-batch-size", type=int, default=512,
                           help="Maximum batch size")
    test_group.add_argument("--batch-step", type=int, default=100,
                           help="Batch size step")
    test_group.add_argument("--seq-len", type=int, default=200,
                           help="Fixed sequence length (tokens)")
    test_group.add_argument("--warmup-rounds", type=int, default=0,
                           help="Number of warmup rounds before measurement")
    test_group.add_argument("--baseline-tokens", type=int, default=14,
                           help="Baseline decode tokens (usually 1)")
    test_group.add_argument("--target-tokens", type=int, default=64,
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
        baseline_tokens: 基准 decode token 数 (通常为 1)
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
                max_tokens=target_tokens, 
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


def main(args: dict):
    # Pop test parameters
    dataset_path = args.pop("dataset_path")
    min_batch_size = args.pop("min_batch_size")
    max_batch_size = args.pop("max_batch_size")
    batch_step = args.pop("batch_step")
    seq_len = args.pop("seq_len")
    warmup_rounds = args.pop("warmup_rounds")
    baseline_tokens = args.pop("baseline_tokens")
    target_tokens = args.pop("target_tokens")
    seed = args.pop("seed_r")
    
    random.seed(seed)
    
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
    
    # Generate batch sizes to test
    batch_sizes = list(range(min_batch_size, max_batch_size + 1, batch_step))
    if max_batch_size not in batch_sizes:
        batch_sizes.append(max_batch_size)
    batch_sizes = sorted(batch_sizes)

    # batch_sizes = [35]  # 临时指定批次大小，快速验证功能
    
    print(f"\n{'='*80}")
    print("Test Configuration")
    print(f"{'='*80}")
    print(f"Number of tests      : {len(batch_sizes)}")
    print(f"Batch sizes          : {batch_sizes}")
    print(f"Sequence length      : {seq_len} tokens")
    print(f"Baseline tokens      : {baseline_tokens}")
    print(f"Target tokens        : {target_tokens}")
    print(f"Decode steps measured: {target_tokens - baseline_tokens}")
    print(f"Warmup rounds        : {warmup_rounds}")
    print(f"Method               : Differential (Time[{target_tokens}] - Time[{baseline_tokens}])")
    
    # Run tests
    results = []
    for i, batch_size in enumerate(batch_sizes):
        # 清理之前的 LLM 实例和 GPU 资源
        del llm
        print("\nCleaning up GPU resources...")
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        time.sleep(2)  # 等待资源释放

        print(f"\nCreating new LLM instance for Batch Size: {batch_size}...")
        
        llm = LLM(**args)

        if dataset is not None:
            prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len)
        else:
            base_text = "This is a test prompt. " * (seq_len // 5)
            prompts = [f"{base_text}\n\nSummarize:" for _ in range(batch_size)]
        
        result = measure_decode_time(
            llm=llm,
            prompts=prompts,
            warmup_rounds=warmup_rounds,
            baseline_tokens=baseline_tokens,
            target_tokens=target_tokens,
            test_id=i + 1,
            batch_size=batch_size,
            seq_len=seq_len
        )
        results.append(result)
    
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
    print(f"\n{'='*110}")
    print("Detailed Results")
    print(f"{'='*110}")
    print(f"{'Batch':<8} {'SeqLen':<8} {'TTFT(s)':<10} {'Decode(s)':<10} {'Steps':<8} {'TPOT(ms)':<12} {'Throughput':<14} {'Status':<8}")
    print("-" * 110)
    
    for r in results:
        status = "✅" if r.get('success', False) else "❌"
        
        if r.get('success', False):
            ttft = r.get('ttft_approx', 0.0)
            pure_time = r.get('pure_decode_time', 0.0)
            steps = r.get('decode_steps', 0)
            tpot = r.get('tpot', 0.0) * 1000  # ms
            throughput = r.get('throughput', 0.0)
            
            print(f"{r['batch_size']:<8} "
                  f"{r['seq_len']:<8} "
                  f"{ttft:<10.4f} "
                  f"{pure_time:<10.4f} "
                  f"{steps:<8} "
                  f"{tpot:<12.2f} "
                  f"{throughput:<14.2f} "
                  f"{status:<8}")
        else:
            print(f"{r['batch_size']:<8} "
                  f"{r.get('seq_len', '-'):<8} "
                  f"{'-':<10} {'-':<10} {'-':<8} {'-':<12} {'-':<14} "
                  f"{status:<8}")
    
    # 保存结果到 JSON
    import json
    output_file = "decode_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*110}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)