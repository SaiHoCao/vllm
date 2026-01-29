# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
import time
import json
import hashlib
import os
from typing import Optional, List, Dict
import torch

from datasets import load_from_disk
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def save_outputs(outputs, mode_name: str, batch_size: int, 
                save_dir: str = "./mode_outputs") -> List[Dict]:
    """保存输出用于对比（包含 batch_size 信息）"""
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    for i, output in enumerate(outputs):
        results.append({
            "id": i,
            "prompt_tokens": len(output.prompt_token_ids),
            "output_token_ids": output.outputs[0].token_ids,
            "output_text": output.outputs[0].text,
        })
    
    filename = f"{mode_name}_bs{batch_size}.json"
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved to {filepath}")
    return results


def compare_outputs(baseline: List[Dict], other: List[Dict], 
                   baseline_name: str, other_name: str) -> bool:
    """详细比较两组输出"""
    print(f"\n{'='*70}")
    print(f"Comparing: {baseline_name} (baseline) vs {other_name}")
    print(f"{'='*70}")
    
    if len(baseline) != len(other):
        print(f"❌ FAIL: Different counts ({len(baseline)} vs {len(other)})")
        return False
    
    mismatches = []
    for i, (b, o) in enumerate(zip(baseline, other)):
        b_tokens = b['output_token_ids']
        o_tokens = o['output_token_ids']
        
        if b_tokens != o_tokens:
            # 找到第一个差异
            diff_idx = 0
            for j, (bt, ot) in enumerate(zip(b_tokens, o_tokens)):
                if bt != ot:
                    diff_idx = j
                    break
            else:
                diff_idx = min(len(b_tokens), len(o_tokens))
            
            mismatches.append({
                'id': i,
                'diff_at': diff_idx,
                'baseline_len': len(b_tokens),
                'other_len': len(o_tokens),
                'baseline_context': b_tokens[max(0,diff_idx-2):diff_idx+3],
                'other_context': o_tokens[max(0,diff_idx-2):diff_idx+3],
            })
    
    if not mismatches:
        # 计算哈希确认
        b_hash = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
        o_hash = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()
        
        print(f"✅ PASS: All {len(baseline)} outputs match perfectly!")
        print(f"   Baseline hash: {b_hash[:32]}")
        print(f"   Other hash:    {o_hash[:32]}")
        return True
    else:
        print(f"❌ FAIL: {len(mismatches)}/{len(baseline)} outputs differ")
        print(f"\nFirst 3 mismatches:")
        for m in mismatches[:3]:
            print(f"  Sample {m['id']}: first diff at token {m['diff_at']}")
            print(f"    {baseline_name}: len={m['baseline_len']}, context={m['baseline_context']}")
            print(f"    {other_name}:    len={m['other_len']}, context={m['other_context']}")
        return False


def create_parser():
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    
    cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)]
    # []
    
    compilation_config = {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
        "replay_mode": "DUAL_PARALLEL",  # 默认模式，会被多模式测试覆盖
    }
    parser.set_defaults(compilation_config=compilation_config)
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=8192)
    parser.set_defaults(enable_chunked_prefill=False)
    
    # Test parameters
    test_group = parser.add_argument_group("Test parameters")
    test_group.add_argument("--dataset-path", type=str, 
                           default="/home/csh/data/projects/datasets/LongBench-v2",
                           help="Path to LongBench-v2 dataset")
    test_group.add_argument("--min-batch-size", type=int, default=100,
                           help="Minimum batch size")
    test_group.add_argument("--max-batch-size", type=int, default=512,
                           help="Maximum batch size")
    test_group.add_argument("--min-seq-len", type=int, default=1024,
                           help="Minimum sequence length (tokens)")
    test_group.add_argument("--max-seq-len", type=int, default=2048,
                           help="Maximum sequence length (tokens)")
    test_group.add_argument("--max-tokens", type=int, default=256,
                           help="Maximum generation length")
    test_group.add_argument("--seed_r", type=int, default=42,
                           help="Random seed")
    test_group.add_argument("--enable-mode-comparison", action="store_true",
                           help="Enable multi-mode comparison test")

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
        prompt = f"{adjusted_text}\n\nPlease summarize the above text briefly:"
        prompts.append(prompt)
    
    return prompts


def run_test_case(llm: LLM, prompts: list[str], 
                  sampling_params: SamplingParams,
                  test_id: int, batch_size: int, 
                  target_seq_len: int,
                  mode_name: str = None,
                  save_output: bool = False) -> dict:
    """运行单个测试用例并返回结果"""
    print(f"\n{'='*70}")
    print(f"Test Case {test_id}")
    print(f"{'='*70}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target sequence length: {target_seq_len} tokens")
    print(f"  Number of prompts: {len(prompts)}")
    if mode_name:
        print(f"  Mode: {mode_name}")
    
    start_time = time.perf_counter()

    try:
        outputs = llm.generate(prompts, sampling_params)
        end_time = time.perf_counter()
    
        elapsed_time = end_time - start_time
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "actual_input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "elapsed_time": elapsed_time,
            "throughput_input": total_input_tokens / elapsed_time,
            "throughput_output": total_output_tokens / elapsed_time,
            "success": True,
            "error": None
        }
        
        # 保存输出用于对比
        if save_output and mode_name:
            saved_results = save_outputs(outputs, mode_name, batch_size)
            result['saved_results'] = saved_results
        
        print(f"  ✅ Success!")
        print(f"  Actual input tokens: {total_input_tokens}")
        print(f"  Output tokens: {total_output_tokens}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Input throughput: {result['throughput_input']:.2f} tokens/s")
        print(f"  Output throughput: {result['throughput_output']:.2f} tokens/s")
        
        # 打印部分输出示例
        print(f"\n  Sample outputs:")
        for i, output in enumerate(outputs[:5]):
            generated_text = output.outputs[0].text.strip().replace("\n", " ")[:80]
            print(f"    [{i}] {generated_text}...")
            
    except Exception as e:
        end_time = time.perf_counter()
        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "actual_input_tokens": 0,
            "output_tokens": 0,
            "elapsed_time": end_time - start_time,
            "throughput_input": 0,
            "throughput_output": 0,
            "success": False,
            "error": str(e)
        }
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    return result


def main(args: dict):
    # Pop test parameters
    dataset_path = args.pop("dataset_path")
    min_batch_size = args.pop("min_batch_size")
    max_batch_size = args.pop("max_batch_size")
    min_seq_len = args.pop("min_seq_len")
    max_seq_len = args.pop("max_seq_len")
    max_tokens = args.pop("max_tokens")
    seed = args.pop("seed_r")
    enable_mode_comparison = args.pop("enable_mode_comparison", False)
    
    # 多模式对比测试配置
    modes_to_test = ["DUAL_PARALLEL", "DUAL_SERIAL", "PADDING"]
    
    # Set random seed
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
        print(f"Dataset columns: {dataset.column_names}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Using fallback synthetic prompts...")
        dataset = None
    
    # Create sampling params (确定性采样)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        ignore_eos=True,
        seed=seed,
    )
    
    # Generate test configurations
    batch_sizes = [450, 100]  # 测试两个不同的 batch size
    
    test_configs = []
    for batch_size in batch_sizes:
        seq_len = 200  # 固定 seq_len
        test_configs.append((batch_size, seq_len))
    
    print(f"\n{'='*70}")
    print("Test Plan")
    print(f"{'='*70}")
    print(f"Number of tests: {len(test_configs)}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Sequence length: fixed at 200")
    print(f"Max generation tokens: {max_tokens}")
    if enable_mode_comparison:
        print(f"Mode comparison enabled: {modes_to_test}")
    print(f"\nTest configurations:")
    for i, (bs, sl) in enumerate(test_configs):
        print(f"  Test {i+1}: batch_size={bs}, seq_len={sl}")
    
    if enable_mode_comparison:
        # 多模式对比测试
        print(f"\n{'#'*70}")
        print(f"# MULTI-MODE COMPARISON TEST")
        print(f"# Modes: {modes_to_test}")
        print(f"{'#'*70}")
        
        all_mode_results = {}  # {mode: {batch_size: results}}
        
        for mode in modes_to_test:
            print(f"\n{'#'*70}")
            print(f"# Testing mode: {mode}")
            print(f"{'#'*70}")
            
            
            print(f"\nInitializing LLM with {mode} mode...")
            cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)]

            llm = LLM(
            model="/home/csh/data/Qwen3-4B",
            max_model_len=8192,
            compilation_config={
                "level": "3",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "replay_mode": mode,
                "cudagraph_capture_sizes": cudagraph_sizes,
            },
            enable_chunked_prefill=False,
        )
            
            mode_results = {}
            
            for i, (batch_size, seq_len) in enumerate(test_configs):
                # # 固定随机种子，确保每个模式使用相同的 prompts
                # random.seed(seed + i)
                
                if dataset is not None:
                    prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len)
                else:
                    base_text = "This is a test prompt. " * (seq_len // 5)
                    prompts = [f"{base_text}\n\nSummarize:" for _ in range(batch_size)]
                
                result = run_test_case(
                    llm=llm,
                    prompts=prompts,
                    sampling_params=sampling_params,
                    test_id=i + 1,
                    batch_size=batch_size,
                    target_seq_len=seq_len,
                    mode_name=mode,
                    save_output=True,
                )
                
                if batch_size not in mode_results:
                    mode_results[batch_size] = []
                mode_results[batch_size].append(result)
            
            all_mode_results[mode] = mode_results
            
            # 清理
            del llm
            torch.cuda.empty_cache()
        
        # 交叉验证所有模式
        print(f"\n{'#'*70}")
        print(f"# CROSS-MODE VALIDATION")
        print(f"{'#'*70}")
        
        baseline_mode = modes_to_test[0]
        validation_results = {}
        
        for batch_size in sorted(set(bs for bs, _ in test_configs)):
            print(f"\n{'='*70}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*70}")
            
            batch_validation = {}
            
            for other_mode in modes_to_test[1:]:
                baseline_data = all_mode_results[baseline_mode].get(batch_size, [])
                other_data = all_mode_results[other_mode].get(batch_size, [])
                
                if baseline_data and other_data:
                    baseline_saved = baseline_data[0].get('saved_results', [])
                    other_saved = other_data[0].get('saved_results', [])
                    
                    if baseline_saved and other_saved:
                        passed = compare_outputs(
                            baseline_saved,
                            other_saved,
                            f"{baseline_mode}_bs{batch_size}",
                            f"{other_mode}_bs{batch_size}"
                        )
                        batch_validation[other_mode] = passed
            
            validation_results[batch_size] = batch_validation
        
        # 汇总验证结果
        print(f"\n{'#'*70}")
        print(f"# VALIDATION SUMMARY")
        print(f"{'#'*70}")
        
        all_passed = True
        for batch_size, validations in validation_results.items():
            print(f"\nBatch Size {batch_size}:")
            for mode, passed in validations.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {baseline_mode} vs {mode}: {status}")
                all_passed = all_passed and passed
        
        print(f"\n{'='*70}")
        if all_passed:
            print("🎉 ALL MODES PRODUCE IDENTICAL OUTPUTS ACROSS ALL BATCH SIZES!")
        else:
            print("⚠️  SOME MODES PRODUCE DIFFERENT OUTPUTS!")
        print(f"{'='*70}")
        
        # 使用第一个模式的结果用于统计
        results = []
        for batch_results in all_mode_results[baseline_mode].values():
            results.extend(batch_results)
    
    else:
        # 单模式测试（原有逻辑）
        print("\nInitializing LLM...")
        llm = LLM(**args)
        
        results = []
        for i, (batch_size, seq_len) in enumerate(test_configs):
            if dataset is not None:
                prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len)
            else:
                base_text = "This is a test prompt. " * (seq_len // 5)
                prompts = [f"{base_text}\n\nSummarize:" for _ in range(batch_size)]
            
            result = run_test_case(
                llm=llm,
                prompts=prompts,
                sampling_params=sampling_params,
                test_id=i + 1,
                batch_size=batch_size,
                target_seq_len=seq_len
            )
            results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_input_throughput = sum(r['throughput_input'] for r in successful) / len(successful)
        avg_output_throughput = sum(r['throughput_output'] for r in successful) / len(successful)
        print(f"\nAverage input throughput: {avg_input_throughput:.2f} tokens/s")
        print(f"\nAverage output throughput: {avg_output_throughput:.2f} tokens/s")
    
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  Test {r['test_id']}: {r['error']}")
    
    # Detailed results table
    print(f"\n{'='*70}")
    print("Detailed Results")
    print(f"{'='*70}")
    print(f"{'Test':<6} {'Batch':<8} {'SeqLen':<8} {'InToks':<10} {'OutToks':<10} {'Time':<8} {'Status':<8}")
    print("-" * 70)
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"{r['test_id']:<6} {r['batch_size']:<8} {r['target_seq_len']:<8} "
              f"{r['actual_input_tokens']:<10} {r['output_tokens']:<10} "
              f"{r['elapsed_time']:<8.2f} {status:<8}")


if __name__ == "__main__":
    parser = create_parser()
    args: dict = vars(parser.parse_args())
    main(args)
