# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
绘图脚本：读取 benchmark 结果并生成对比图
运行方式: python plot_results.py benchmark_results/20241215_xxxx
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(output_dir: str) -> dict:
    """加载所有配置的结果"""
    results = {}
    
    # 加载元数据
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # 加载每个配置的结果
    for filename in os.listdir(output_dir):
        if filename.startswith("results_") and filename.endswith(".json"):
            config_name = filename[8:-5]  # 去掉 "results_" 和 ".json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'r') as f:
                results[config_name] = json.load(f)
    
    return metadata, results


def plot_throughput_vs_batch_size(metadata: dict, results: dict, output_dir: str):
    """绘制吞吐量 vs batch size"""
    plt.figure(figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for idx, (config_name, data) in enumerate(results.items()):
        successful = [r for r in data["results"] if r["success"]]
        if not successful:
            continue
        
        batch_sizes = [r["batch_size"] for r in successful]
        throughputs = [r["throughput_output"] for r in successful]
        
        plt.scatter(batch_sizes, throughputs, 
                   label=data["config_description"],
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   alpha=0.7, s=20)
    
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Output Throughput (tokens/s)", fontsize=12)
    plt.title("Output Throughput vs Batch Size", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "throughput_vs_batch_size.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_latency_vs_batch_size(metadata: dict, results: dict, output_dir: str):
    """绘制延迟 vs batch size"""
    plt.figure(figsize=(14, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for idx, (config_name, data) in enumerate(results.items()):
        successful = [r for r in data["results"] if r["success"]]
        if not successful:
            continue
        
        batch_sizes = [r["batch_size"] for r in successful]
        latencies = [r["elapsed_time"] for r in successful]
        
        plt.scatter(batch_sizes, latencies, 
                   label=data["config_description"],
                   color=colors[idx % len(colors)],
                   marker=markers[idx % len(markers)],
                   alpha=0.7, s=20)
    
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Total Latency (s)", fontsize=12)
    plt.title("Total Latency vs Batch Size", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "latency_vs_batch_size.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_speedup_comparison(metadata: dict, results: dict, output_dir: str):
    """绘制相对于 baseline 的加速比"""
    if "baseline_no_cudagraph" not in results:
        print("Warning: baseline_no_cudagraph not found, skipping speedup plot")
        return
    
    baseline_data = results["baseline_no_cudagraph"]
    baseline_times = {r["batch_size"]: r["elapsed_time"] 
                      for r in baseline_data["results"] if r["success"]}
    
    plt.figure(figsize=(14, 6))
    
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    markers = ['s', '^', 'D']
    
    idx = 0
    for config_name, data in results.items():
        if config_name == "baseline_no_cudagraph":
            continue
        
        successful = [r for r in data["results"] if r["success"]]
        if not successful:
            continue
        
        batch_sizes = []
        speedups = []
        
        for r in successful:
            bs = r["batch_size"]
            if bs in baseline_times and baseline_times[bs] > 0:
                speedup = baseline_times[bs] / r["elapsed_time"]
                batch_sizes.append(bs)
                speedups.append(speedup)
        
        if batch_sizes:
            plt.scatter(batch_sizes, speedups, 
                       label=data["config_description"],
                       color=colors[idx % len(colors)],
                       marker=markers[idx % len(markers)],
                       alpha=0.7, s=20)
        idx += 1
    
    plt.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1.0x)')
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Speedup vs Baseline", fontsize=12)
    plt.title("Speedup Comparison (vs No CUDA Graph)", fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "speedup_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_summary_bar_chart(metadata: dict, results: dict, output_dir: str):
    """绘制汇总柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    config_names = []
    avg_throughputs = []
    total_times = []
    
    for config_name, data in results.items():
        if "summary" in data:
            config_names.append(data["config_description"].replace("CUDA Graph: ", "").replace("Baseline: ", ""))
            avg_throughputs.append(data["summary"]["avg_throughput_output"])
            total_times.append(data["summary"]["total_time"])
    
    x = np.arange(len(config_names))
    width = 0.6
    
    # 平均吞吐量
    bars1 = axes[0].bar(x, avg_throughputs, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(config_names)])
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Avg Output Throughput (tokens/s)")
    axes[0].set_title("Average Output Throughput")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(config_names, rotation=15, ha='right')
    axes[0].bar_label(bars1, fmt='%.1f')
    
    # 总时间
    bars2 = axes[1].bar(x, total_times, width, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(config_names)])
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel("Total Time (s)")
    axes[1].set_title("Total Benchmark Time")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(config_names, rotation=15, ha='right')
    axes[1].bar_label(bars2, fmt='%.1f')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "summary_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <output_dir>")
        print("Example: python plot_results.py benchmark_results/20241215_123456")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist")
        sys.exit(1)
    
    print(f"Loading results from {output_dir}...")
    metadata, results = load_results(output_dir)
    
    print(f"Found {len(results)} configurations")
    
    # 生成各种图表
    plot_throughput_vs_batch_size(metadata, results, output_dir)
    plot_latency_vs_batch_size(metadata, results, output_dir)
    plot_speedup_comparison(metadata, results, output_dir)
    plot_summary_bar_chart(metadata, results, output_dir)
    
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()