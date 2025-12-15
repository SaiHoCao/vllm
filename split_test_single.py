# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
单配置测试脚本：被 benchmark_configs.py 调用
"""

import json
import random
import time
from typing import Optional

from datasets import load_from_disk
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    
    parser.add_argument("--test-configs-file", type=str, required=True,
                       help="Path to test configs JSON file")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Path to output results JSON file")
    parser.add_argument("--config-name", type=str, required=True,
                       help="Configuration name")
    parser.add_argument("--config-description", type=str, default="",
                       help="Configuration description")
    parser.add_argument("--dataset-path", type=str, 
                       default="../../datasets/LongBench-v2",
                       help="Path to dataset")
    parser.add_argument("--max-tokens", type=int, default=128,
                       help="Maximum generation length")
    parser.add_argument("--seed_r", type=int, default=42,
                       help="Random seed for prompt selection")
    
    return parser


def truncate_or_pad_text(text: str, target_tokens: int, 
                         avg_chars_per_token: float = 4.0) -> str:
    target_chars = int(target_tokens * avg_chars_per_token)
    if len(text) >= target_chars:
        return text[:target_chars]
    else:
        repeated = text * (target_chars // len(text) + 1)
        return repeated[:target_chars]


def prepare_prompts_from_dataset(dataset, batch_size: int, 
                                  target_seq_len: int, seed: int) -> list[str]:
    """使用固定 seed 确保相同的 prompt 选择"""
    random.seed(seed)
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
                  target_seq_len: int) -> dict:
    
    print(f"  Test {test_id}: batch_size={batch_size}, seq_len={target_seq_len}", end=" ")
    
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
            "throughput_total": (total_input_tokens + total_output_tokens) / elapsed_time,
            "throughput_output": total_output_tokens / elapsed_time,
            "latency_per_output_token": elapsed_time / total_output_tokens if total_output_tokens > 0 else 0,
            "success": True,
            "error": None
        }
        
        print(f"✅ {elapsed_time:.2f}s, {result['throughput_output']:.1f} tok/s")
        
    except Exception as e:
        end_time = time.perf_counter()
        result = {
            "test_id": test_id,
            "batch_size": batch_size,
            "target_seq_len": target_seq_len,
            "actual_input_tokens": 0,
            "output_tokens": 0,
            "elapsed_time": end_time - start_time,
            "throughput_total": 0,
            "throughput_output": 0,
            "latency_per_output_token": 0,
            "success": False,
            "error": str(e)
        }
        print(f"❌ {e}")
    
    return result


def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    
    # 提取非 LLM 参数
    test_configs_file = args.pop("test_configs_file")
    output_file = args.pop("output_file")
    config_name = args.pop("config_name")
    config_description = args.pop("config_description")
    dataset_path = args.pop("dataset_path")
    max_tokens = args.pop("max_tokens")
    seed = args.pop("seed_r")
    
    # 加载测试配置
    with open(test_configs_file, 'r') as f:
        test_configs = [tuple(x) for x in json.load(f)]
    
    print(f"Config: {config_name}")
    print(f"Description: {config_description}")
    print(f"Number of test cases: {len(test_configs)}")
    
    # 加载数据集
    print(f"Loading dataset from {dataset_path}...")
    try:
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, 'keys'):
            split_name = list(dataset.keys())[0]
            dataset = dataset[split_name]
        print(f"Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        dataset = None
    
    # 创建 LLM
    print("Initializing LLM...")
    llm = LLM(**args)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )
    
    # 运行测试
    print(f"\nRunning {len(test_configs)} tests...")
    results = []
    
    for i, (batch_size, seq_len) in enumerate(test_configs):
        # 使用固定 seed 确保相同的 prompt
        prompt_seed = seed + i
        
        if dataset is not None:
            prompts = prepare_prompts_from_dataset(dataset, batch_size, seq_len, prompt_seed)
        else:
            random.seed(prompt_seed)
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
    
    # 保存结果
    output_data = {
        "config_name": config_name,
        "config_description": config_description,
        "num_tests": len(results),
        "num_successful": sum(1 for r in results if r["success"]),
        "results": results,
    }
    
    # 计算汇总统计
    successful = [r for r in results if r["success"]]
    if successful:
        output_data["summary"] = {
            "avg_throughput_output": sum(r["throughput_output"] for r in successful) / len(successful),
            "avg_latency_per_token": sum(r["latency_per_output_token"] for r in successful) / len(successful),
            "total_time": sum(r["elapsed_time"] for r in successful),
        }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Successful: {output_data['num_successful']}/{output_data['num_tests']}")


if __name__ == "__main__":
    main()