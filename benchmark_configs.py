# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
主脚本：协调不同配置的测试
运行方式: python benchmark_configs.py
"""

import subprocess
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# 定义要测试的配置
CONFIGS = {
    "baseline_no_cudagraph": {
        "level": "0",
        "cudagraph_mode": "NONE",
        "enable_cudagraph_split": False,
        "enable_split_parallel_streams": False,
        "description": "Baseline: No CUDA Graph"
    },
    "cudagraph_no_split": {
        "level": "0",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "enable_cudagraph_split": False,
        "enable_split_parallel_streams": False,
        "description": "CUDA Graph: No Split (Padding)"
    },
    "cudagraph_split_serial": {
        "level": "0",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "enable_cudagraph_split": True,
        "enable_split_parallel_streams": False,
        "description": "CUDA Graph: Split Serial"
    },
    "cudagraph_split_parallel": {
        "level": "0",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "enable_cudagraph_split": True,
        "enable_split_parallel_streams": True,
        "description": "CUDA Graph: Split Parallel Streams"
    },
}


def generate_test_configs(min_batch: int = 1, max_batch: int = 512,
                          min_seq: int = 1024, max_seq: int = 2048,
                          seed: int = 42) -> list[tuple[int, int]]:
    """
    生成固定的测试配置（batch_size, seq_len）对。
    使用固定的 seed 确保每次运行相同。
    """
    import random
    random.seed(seed)
    
    batch_sizes = []
    if min_batch <= 8:
        batch_sizes.append(1)
        batch_sizes.append(2)
        batch_sizes.append(4)
    
    start = max(8, min_batch if min_batch % 2 == 0 else min_batch + 1)

    # 以步长 8 生成 batch sizes
    for bs in range(start, max_batch + 1, 8):
        batch_sizes.append(bs)
    
    if max_batch not in batch_sizes and max_batch >= min_batch:
        batch_sizes.append(max_batch)
    
    batch_sizes = sorted(set(bs for bs in batch_sizes if min_batch <= bs <= max_batch))
    
    # 为每个 batch size 生成固定的 seq_len
    test_configs = []
    for batch_size in batch_sizes:
        seq_len = random.randint(min_seq, max_seq)
        test_configs.append((batch_size, seq_len))
    
    return test_configs


def save_test_configs(configs: list[tuple[int, int]], filepath: str):
    """保存测试配置到文件"""
    with open(filepath, 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"Saved {len(configs)} test configs to {filepath}")


def load_test_configs(filepath: str) -> list[tuple[int, int]]:
    """从文件加载测试配置"""
    with open(filepath, 'r') as f:
        return [tuple(x) for x in json.load(f)]


def run_single_config(config_name: str, config: dict, 
                      test_configs_file: str, output_dir: str,
                      model_path: str, max_tokens: int = 128,
                      dataset_path: str = "../../datasets/LongBench-v2"):
    """运行单个配置的测试"""
    
    output_file = os.path.join(output_dir, f"results_{config_name}.json")
    
    # 构建命令
    compilation_config_str = json.dumps({
        "level": config["level"],
        "cudagraph_mode": config["cudagraph_mode"],
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)], # 1 到 512
        "enable_cudagraph_split": config["enable_cudagraph_split"],
        "enable_split_parallel_streams": config["enable_split_parallel_streams"],
    })
    
    cmd = [
        sys.executable, "split_test_single.py",
        "--model", model_path,
        "--max-model-len", "8192",
        "--compilation-config", compilation_config_str,
        "--test-configs-file", test_configs_file,
        "--output-file", output_file,
        "--config-name", config_name,
        "--config-description", config["description"],
        "--dataset-path", dataset_path,
        "--max-tokens", str(max_tokens),
    ]
    
    print(f"\n{'='*70}")
    print(f"Running config: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*70}")
    
    # 运行子进程
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"❌ Config {config_name} failed with return code {result.returncode}")
        return False
    
    print(f"✅ Config {config_name} completed successfully")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark different CUDA Graph configurations")
    parser.add_argument("--model", type=str, default="/home/csh/data/Qwen3-0.6B",
                       help="Model path")
    parser.add_argument("--dataset-path", type=str, default="../../datasets/LongBench-v2",
                       help="Dataset path")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--min-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-size", type=int, default=512)
    parser.add_argument("--min-seq-len", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", type=str, nargs="+", 
                       choices=list(CONFIGS.keys()) + ["all"],
                       default=["all"],
                       help="Which configs to run")
    args = parser.parse_args()
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成并保存固定的测试配置
    test_configs_file = os.path.join(output_dir, "test_configs.json")
    test_configs = generate_test_configs(
        args.min_batch_size, args.max_batch_size,
        args.min_seq_len, args.max_seq_len,
        args.seed
    )
    save_test_configs(test_configs, test_configs_file)
    
    # 确定要运行的配置
    configs_to_run = CONFIGS if "all" in args.configs else {k: CONFIGS[k] for k in args.configs}
    
    print(f"\n{'='*70}")
    print("Benchmark Configuration")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Number of test cases: {len(test_configs)}")
    print(f"Batch size range: [{args.min_batch_size}, {args.max_batch_size}]")
    print(f"Sequence length range: [{args.min_seq_len}, {args.max_seq_len}]")
    print(f"Configs to run: {list(configs_to_run.keys())}")
    
    # 保存元数据
    metadata = {
        "timestamp": timestamp,
        "model": args.model,
        "min_batch_size": args.min_batch_size,
        "max_batch_size": args.max_batch_size,
        "min_seq_len": args.min_seq_len,
        "max_seq_len": args.max_seq_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "num_test_cases": len(test_configs),
        "configs": {k: v["description"] for k, v in configs_to_run.items()},
    }
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=8)
    
    # 运行每个配置
    results_summary = {}
    for config_name, config in configs_to_run.items():
        success = run_single_config(
            config_name=config_name,
            config=config,
            test_configs_file=test_configs_file,
            output_dir=output_dir,
            model_path=args.model,
            max_tokens=args.max_tokens,
            dataset_path=args.dataset_path,
        )
        results_summary[config_name] = "success" if success else "failed"
    
    # 打印总结
    print(f"\n{'='*70}")
    print("Benchmark Complete")
    print(f"{'='*70}")
    for config_name, status in results_summary.items():
        print(f"  {config_name}: {status}")
    print(f"\nResults saved to: {output_dir}")
    print(f"Run 'python plot_results.py {output_dir}' to generate plots")


if __name__ == "__main__":
    main()