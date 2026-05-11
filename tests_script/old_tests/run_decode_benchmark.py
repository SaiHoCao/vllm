# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
自动化 Benchmark 测试脚本
针对不同 CUDA Graph 配置进行性能测试
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# 测试配置
CONFIGS = {
    "baseline_no_cudagraph": {
        "level": "3",
        "cudagraph_mode": "NONE",
        "enable_cudagraph_split": False,
        "enable_dual_graph": False,
        "description": "Baseline: No CUDA Graph"
    },
    "cudagraph_padding": {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "enable_cudagraph_split": False,
        "enable_dual_graph": False,
        "description": "CUDA Graph: Padding"
    },
    "cudagraph_dual_stream": {
        "level": "3",
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "enable_cudagraph_split": True,
        "enable_dual_graph": True,
        "description": "CUDA Graph: Dual Stream Replay Execution"
    },
}

# CUDA Graph 大小配置
CUDAGRAPH_SIZES = [1, 2, 4, 8, 16, 32] + [i * 64 for i in range(1, 9)]


def run_single_config(config_name: str, config: dict, args: argparse.Namespace):
    """运行单个配置的测试"""
    
    print(f"\n{'='*80}")
    print(f"Running Configuration: {config_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*80}")
    
    # 构建命令行参数
    cmd = [
        "python", "tests_script/decode_test_diff.py",
        f"--max-model-len={args.max_model_len}",
        f"--dataset-path={args.dataset_path}",
        f"--min-batch-size={args.min_batch_size}",
        f"--max-batch-size={args.max_batch_size}",
        f"--batch-step={args.batch_step}",
        f"--seq-len={args.seq_len}",
        f"--warmup-rounds={args.warmup_rounds}",
        f"--baseline-tokens={args.baseline_tokens}",
        f"--target-tokens={args.target_tokens}",
    ]
    
    # 添加编译配置
    compilation_config = {
        "level": config["level"],
        "cudagraph_mode": config["cudagraph_mode"],
        "enable_cudagraph_split": config["enable_cudagraph_split"],
        "enable_dual_graph": config["enable_dual_graph"],
    }
    
    # 如果使用 CUDA Graph,添加 capture sizes
    if config["cudagraph_mode"] != "NONE":
        compilation_config["cudagraph_capture_sizes"] = CUDAGRAPH_SIZES
    
    cmd.append(f"--compilation-config={json.dumps(compilation_config)}")
    
    # 运行测试
    start_time = time.time()
    try:
        result = subprocess.run(cmd, text=True, check=True)
        # print(result.stdout)
        success = True
        error_msg = None
    except subprocess.CalledProcessError as e:
        print(f"Error running config {config_name}:")
        print(e.stdout)
        print(e.stderr)
        success = False
        error_msg = str(e)
    
    elapsed_time = time.time() - start_time
    
    # 读取生成的结果文件
    results_file = "decode_benchmark_results.json"
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 重命名并保存配置特定的结果
        config_results_file = f"results_{config_name}.json"
        os.rename(results_file, config_results_file)
        print(f"Results saved to {config_results_file}")
    else:
        results = []
        config_results_file = None
    
    return {
        "config_name": config_name,
        "config": config,
        "success": success,
        "error": error_msg,
        "elapsed_time": elapsed_time,
        "results": results,
        "results_file": config_results_file
    }


def merge_results(all_results: list):
    """合并所有配置的测试结果"""
    
    merged = {
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "batch_sizes": set()
    }
    
    for result in all_results:
        config_name = result["config_name"]
        merged["configs"][config_name] = {
            "description": result["config"]["description"],
            "success": result["success"],
            "elapsed_time": result["elapsed_time"],
            "results": []
        }
        
        if result["success"] and result["results"]:
            for r in result["results"]:
                merged["configs"][config_name]["results"].append(r)
                merged["batch_sizes"].add(r["batch_size"])
    
    merged["batch_sizes"] = sorted(list(merged["batch_sizes"]))
    
    return merged


def create_comparison_data(merged_results: dict):
    """创建用于绘图的比较数据"""
    
    comparison_data = {
        "batch_sizes": merged_results["batch_sizes"],
        "metrics": {
            "ttft": {},
            "pure_decode_time": {},
            "tpot": {},
            "throughput": {}
        }
    }
    
    for config_name, config_data in merged_results["configs"].items():
        if not config_data["success"]:
            continue
        
        # 按 batch_size 组织数据
        batch_metrics = {}
        for r in config_data["results"]:
            if not r.get("success", False):
                continue
            
            batch_size = r["batch_size"]
            batch_metrics[batch_size] = {
                "ttft": r.get("ttft_approx", 0),
                "pure_decode_time": r.get("pure_decode_time", 0),
                "tpot": r.get("tpot", 0) * 1000,  # ms
                "throughput": r.get("throughput", 0)
            }
        
        # 填充到对应的 metric 字典
        for metric in comparison_data["metrics"]:
            comparison_data["metrics"][metric][config_name] = []
            for bs in comparison_data["batch_sizes"]:
                if bs in batch_metrics:
                    comparison_data["metrics"][metric][config_name].append(
                        batch_metrics[bs][metric]
                    )
                else:
                    comparison_data["metrics"][metric][config_name].append(None)
    
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks with different CUDA Graph configurations")
    
    # 模型和数据集参数
    parser.add_argument("--model", type=str, default="/home/csh/data/Qwen3-0.6B",
                       help="Model path")
    parser.add_argument("--max-model-len", type=int, default=8192,
                       help="Maximum model length")
    parser.add_argument("--dataset-path", type=str, 
                       default="../../datasets/LongBench-v2",
                       help="Dataset path")
    
    # 测试参数
    parser.add_argument("--min-batch-size", type=int, default=40,
                       help="Minimum batch size")
    parser.add_argument("--max-batch-size", type=int, default=512,
                       help="Maximum batch size")
    parser.add_argument("--batch-step", type=int, default=40,
                       help="Batch size step")
    parser.add_argument("--seq-len", type=int, default=200,
                       help="Sequence length")
    parser.add_argument("--warmup-rounds", type=int, default=0,
                       help="Warmup rounds")
    parser.add_argument("--baseline-tokens", type=int, default=4,
                       help="Baseline decode tokens")
    parser.add_argument("--target-tokens", type=int, default=64,
                       help="Target decode tokens")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # 配置选择
    parser.add_argument("--configs", nargs='+', 
                       choices=list(CONFIGS.keys()) + ['all'],
                       default=['all'],
                       help="Which configurations to test")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="0.6_decode_diff_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 确定要测试的配置
    if 'all' in args.configs:
        configs_to_test = CONFIGS.keys()
    else:
        configs_to_test = args.configs
    
    print(f"\n{'='*80}")
    print("Benchmark Configuration")
    print(f"{'='*80}")
    print(f"Configurations to test: {', '.join(configs_to_test)}")
    print(f"Batch sizes: {args.min_batch_size} to {args.max_batch_size} (step {args.batch_step})")
    print(f"Sequence length: {args.seq_len}")
    print(f"Baseline tokens: {args.baseline_tokens}")
    print(f"Target tokens: {args.target_tokens}")
    print(f"Output directory: {output_dir}")
    
    # 运行所有配置
    all_results = []
    for config_name in configs_to_test:
        config = CONFIGS[config_name]
        result = run_single_config(config_name, config, args)
        all_results.append(result)
        
        # 保存每个配置的独立结果
        config_file = output_dir / f"{config_name}_result.json"
        with open(config_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ {config_name} results saved to {config_file}")
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Benchmark Summary")
    print(f"{'='*80}")
    
    for result in all_results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{result['config_name']:<30} {status:<15} ({result['elapsed_time']:.1f}s)")
    
    print(f"\n{'='*80}")
    print(f"All results saved to {output_dir}")
    print(f"Run 'python merge_decode_diff_results.py --input-dir {output_dir}' to merge results ")
    print(f"and Run 'python plot_decode_diff_results.py --input-dir {output_dir}' to plot.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()