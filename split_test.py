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
    
    cudagraph_sizes = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)] # [1,2,4,8,16,32,64,128,192,256,320,384,448,512]
    
    compilation_config = {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": cudagraph_sizes,
        "enable_cudagraph_split": True,
        "enable_dual_graph": True,
    }
    parser.set_defaults(compilation_config=compilation_config)
    parser.set_defaults(model="/home/csh/data/Qwen3-4B")
    parser.set_defaults(max_model_len=8192)  # 支持长序列
    
    # Test parameters
    test_group = parser.add_argument_group("Test parameters")
    test_group.add_argument("--dataset-path", type=str, 
                           default="../../datasets/LongBench-v2",
                           help="Path to LongBench-v2 dataset")
    test_group.add_argument("--num-tests", type=int, default=10,
                           help="Number of test cases to run")
    test_group.add_argument("--min-batch-size", type=int, default=100,
                           help="Minimum batch size")
    test_group.add_argument("--max-batch-size", type=int, default=512,
                           help="Maximum batch size")
    test_group.add_argument("--min-seq-len", type=int, default=1024,
                           help="Minimum sequence length (tokens)")
    test_group.add_argument("--max-seq-len", type=int, default=2048,
                           help="Maximum sequence length (tokens)")
    test_group.add_argument("--max-tokens", type=int, default=10,
                           help="Maximum generation length")
    test_group.add_argument("--seed_r", type=int, default=42,
                           help="Random seed")

    return parser


def truncate_or_pad_text(text: str, target_tokens: int, 
                         avg_chars_per_token: float = 4.0) -> str:
    """
    截断或重复文本以达到目标 token 数。
    使用粗略估算：1 token ≈ 4 个字符（英文）
    """
    target_chars = int(target_tokens * avg_chars_per_token)
    
    if len(text) >= target_chars:
        # 截断
        return text[:target_chars]
    else:
        # 重复文本直到达到目标长度
        repeated = text * (target_chars // len(text) + 1)
        return repeated[:target_chars]


def prepare_prompts_from_dataset(dataset, batch_size: int, 
                                  target_seq_len: int) -> list[str]:
    """
    从数据集中准备指定数量和长度的 prompts。
    """
    prompts = []
    dataset_size = len(dataset)
    
    for i in range(batch_size):
        # 随机选择一个样本
        idx = random.randint(0, dataset_size - 1)
        sample = dataset[idx]
        
        # 获取文本内容（LongBench-v2 的字段可能是 'context' 或 'input'）
        if 'context' in sample:
            text = sample['context']
        elif 'input' in sample:
            text = sample['input']
        elif 'question' in sample:
            text = sample['question']
        else:
            # 尝试获取第一个字符串字段
            text = str(list(sample.values())[0])
        
        # 调整到目标长度
        adjusted_text = truncate_or_pad_text(text, target_seq_len)
        
        # 添加简单的问题后缀
        prompt = f"{adjusted_text}\n\nPlease summarize the above text briefly:"
        prompts.append(prompt)
    
    return prompts


def run_test_case(llm: LLM, prompts: list[str], 
                  sampling_params: SamplingParams,
                  test_id: int, batch_size: int, 
                  target_seq_len: int) -> dict:
    """
    运行单个测试用例并返回结果。
    """
    print(f"\n{'='*70}")
    print(f"Test Case {test_id}")
    print(f"{'='*70}")
    print(f"  Batch size: {batch_size}")
    print(f"  Target sequence length: {target_seq_len} tokens")
    print(f"  Number of prompts: {len(prompts)}")
    
    # 记录开始时间
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
        
        print(f"  ✅ Success!")
        print(f"  Actual input tokens: {total_input_tokens}")
        print(f"  Output tokens: {total_output_tokens}")
        print(f"  Elapsed time: {elapsed_time:.2f}s")
        print(f"  Input throughput: {result['throughput_input']:.2f} tokens/s")
        print(f"  Output throughput: {result['throughput_output']:.2f} tokens/s")
        
        # 打印部分输出示例
        print(f"\n  Sample outputs:")
        for i, output in enumerate(outputs[:2]):
            generated_text = output.outputs[0].text[:100]
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
    num_tests = args.pop("num_tests")  # 这个参数将不再使用
    min_batch_size = args.pop("min_batch_size")
    max_batch_size = args.pop("max_batch_size")
    min_seq_len = args.pop("min_seq_len")
    max_seq_len = args.pop("max_seq_len")
    max_tokens = args.pop("max_tokens")
    seed = args.pop("seed_r")
    
    # Set random seed
    random.seed(seed)
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        # 如果是 DatasetDict，取第一个 split
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
    
    # Create LLM
    print("\nInitializing LLM...")
    llm = LLM(**args)
    
    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # 使用贪心解码以保证可复现性
    )
    
    # Generate test configurations: 全范围 batch size 测试
    # 1, 2, 4, 6, 8, 10, ..., 510, 512
    test_configs = []
    batch_sizes = []
    
    # # 生成 batch size 列表: 1, 2, 4, 6, 8, ..., 510, 512
    # if min_batch_size <= 1:
    #     batch_sizes.append(1)
    
    # # 从 2 开始，每隔 2 一个
    # start = max(2, min_batch_size if min_batch_size % 2 == 0 else min_batch_size + 1)
    # for bs in range(start, max_batch_size + 1, 2):
    #     batch_sizes.append(bs)
    
    # # 确保 max_batch_size 在列表中
    # if max_batch_size not in batch_sizes and max_batch_size >= min_batch_size:
    #     batch_sizes.append(max_batch_size)
    
    # # 过滤范围
    # batch_sizes = [bs for bs in batch_sizes if min_batch_size <= bs <= max_batch_size]
    # batch_sizes = sorted(set(batch_sizes))  # 去重并排序

    batch_sizes = [336]  # 仅测试两个 batch size，快速验证功能
    
    # 为每个 batch size 随机生成一个 seq_len
    for batch_size in batch_sizes:
        # seq_len = random.randint(min_seq_len, max_seq_len)
        seq_len = 1536  # 固定 seq_len，快速验证功能
        test_configs.append((batch_size, seq_len))
    
    print(f"\n{'='*70}")
    print("Test Plan")
    print(f"{'='*70}")
    print(f"Number of tests: {len(test_configs)}")
    print(f"Batch sizes: {batch_sizes[:10]}...{batch_sizes[-5:]}" if len(batch_sizes) > 15 else f"Batch sizes: {batch_sizes}")
    print(f"Sequence length range: [{min_seq_len}, {max_seq_len}] (random)")
    print(f"Max generation tokens: {max_tokens}")
    print(f"\nTest configurations (first 10 and last 5):")
    configs_to_show = test_configs[:10] + test_configs[-5:] if len(test_configs) > 15 else test_configs
    for i, (bs, sl) in enumerate(configs_to_show[:10]):
        print(f"  Test {i+1}: batch_size={bs}, seq_len={sl}")
    if len(test_configs) > 15:
        print(f"  ... ({len(test_configs) - 15} more tests) ...")
        for i, (bs, sl) in enumerate(test_configs[-5:]):
            print(f"  Test {len(test_configs) - 4 + i}: batch_size={bs}, seq_len={sl}")
    
    # Run tests
    results = []
    for i, (batch_size, seq_len) in enumerate(test_configs):
        if dataset is not None:
            prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len)
        else:
            # Fallback: 使用合成 prompts
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
        print(f"Average output throughput: {avg_output_throughput:.2f} tokens/s")
    
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