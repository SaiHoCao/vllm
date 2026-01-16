# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
绘制 Benchmark 结果对比图
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 配置颜色和样式 - 使用更亮、区分度更大的颜色
STYLE_CONFIG = {
    "baseline_no_cudagraph": {
        "color": "#FF4757",  # 鲜红色
        "linestyle": "-",
        "marker": "o",
        "label": "No CUDA Graph"
    },
    "cudagraph_padding": {
        "color": "#2ED573",  # 鲜绿色
        "linestyle": "-",
        "marker": "s",
        "label": "Padding"
    },
    "cudagraph_mixed_parallel": {
        "color": "#FFA502",  # 鲜橙色
        "linestyle": "-",
        "marker": "^",
        "label": "Mixed Parallel"
    },
    "cudagraph_dual_stream": {
        "color": "#5F27CD",  # 鲜紫色
        "linestyle": "-",
        "marker": "D",
        "label": "Dual Stream"
    },
}

# 指标配置
METRICS_CONFIG = {
    "ttft": {
        "title": "Time To First Token (TTFT)",
        "ylabel": "Time (seconds)",
        "filename": "ttft_comparison.png"
    },
    "pure_decode_time": {
        "title": "Pure Decode Time",
        "ylabel": "Time (seconds)",
        "filename": "decode_time_comparison.png"
    },
    "tpot": {
        "title": "Time Per Output Token (TPOT)",
        "ylabel": "Time (milliseconds)",
        "filename": "tpot_comparison.png"
    },
    "throughput": {
        "title": "Decode Throughput",
        "ylabel": "Tokens/second",
        "filename": "throughput_comparison.png"
    }
}


def plot_metric(batch_sizes, metric_data, metric_config, config_name, output_dir):
    """绘制单个指标的对比图"""
    
    plt.figure(figsize=(12, 7))
    
    for config_name, values in metric_data.items():
        style = STYLE_CONFIG.get(config_name, {})
        
        # 过滤 None 值
        valid_points = [(bs, v) for bs, v in zip(batch_sizes, values) if v is not None]
        if not valid_points:
            continue
        
        valid_batch_sizes, valid_values = zip(*valid_points)
        
        plt.plot(
            valid_batch_sizes,
            valid_values,
            color=style.get("color", "blue"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=2,
            markersize=8,
            label=style.get("label", config_name),
            alpha=0.8
        )
    
    plt.xlabel("Batch Size", fontsize=12, fontweight='bold')
    plt.ylabel(metric_config["ylabel"], fontsize=12, fontweight='bold')
    plt.title(metric_config["title"], fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # 设置 x 轴为整数刻度
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    output_file = output_dir / metric_config["filename"]
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    plt.close()


def plot_speedup(batch_sizes, metric_data, baseline_name, output_dir):
    """绘制相对于 baseline 的加速比"""
    
    if baseline_name not in metric_data:
        print(f"Warning: Baseline '{baseline_name}' not found, skipping speedup plot")
        return
    
    baseline_values = metric_data[baseline_name]
    
    plt.figure(figsize=(12, 7))
    
    for config_name, values in metric_data.items():
        if config_name == baseline_name:
            continue
        
        style = STYLE_CONFIG.get(config_name, {})
        
        # 计算加速比
        speedup = []
        valid_batch_sizes = []
        for bs, baseline, current in zip(batch_sizes, baseline_values, values):
            if baseline is not None and current is not None and baseline > 0:
                speedup.append(baseline / current)
                valid_batch_sizes.append(bs)
        
        if not speedup:
            continue
        
        plt.plot(
            valid_batch_sizes,
            speedup,
            color=style.get("color", "blue"),
            linestyle=style.get("linestyle", "-"),
            marker=style.get("marker", "o"),
            linewidth=2,
            markersize=8,
            label=style.get("label", config_name),
            alpha=0.8
        )
    
    # 添加 y=1 的参考线
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='Baseline')
    
    plt.xlabel("Batch Size", fontsize=12, fontweight='bold')
    plt.ylabel("Speedup (vs No CUDA Graph)", fontsize=12, fontweight='bold')
    plt.title("Throughput Speedup Comparison", fontsize=14, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    
    output_file = output_dir / "speedup_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    
    plt.close()


def create_summary_table(comparison_data, output_dir):
    """创建汇总表格图"""
    
    batch_sizes = comparison_data["batch_sizes"]
    configs = list(comparison_data["metrics"]["throughput"].keys())
    
    # 创建表格数据
    fig, ax = plt.subplots(figsize=(14, max(6, len(batch_sizes) * 0.4)))
    ax.axis('tight')
    ax.axis('off')
    
    # 表头
    headers = ["Batch Size"] + [STYLE_CONFIG.get(c, {}).get("label", c) for c in configs]
    
    # 表格数据
    table_data = []
    for bs in batch_sizes:
        row = [str(bs)]
        for config in configs:
            throughput_values = comparison_data["metrics"]["throughput"][config]
            idx = comparison_data["batch_sizes"].index(bs)
            value = throughput_values[idx]
            if value is not None:
                row.append(f"{value:.1f}")
            else:
                row.append("-")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15] + [0.2] * len(configs))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表头样式 - 使用更亮的颜色
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#1E90FF')  # 鲜亮的蓝色
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 交替行颜色 - 增加对比度
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')  # 浅蓝色
            else:
                table[(i, j)].set_facecolor('#FFFFFF')  # 白色
    
    plt.title("Throughput Summary (tokens/s)", fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / "summary_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved table: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--input-dir", type=str, default="benchmark_decode_results",
                       help="Input directory containing comparison_data.json")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for plots (default: same as input-dir)")
    parser.add_argument("--baseline", type=str, default="baseline_no_cudagraph",
                       help="Baseline configuration for speedup calculation")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(exist_ok=True)
    
    # 读取比较数据
    comparison_file = input_dir / "comparison_data.json"
    if not comparison_file.exists():
        print(f"Error: {comparison_file} not found!")
        return
    
    with open(comparison_file, 'r') as f:
        comparison_data = json.load(f)
    
    print(f"\n{'='*80}")
    print("Generating Plots")
    print(f"{'='*80}")
    
    batch_sizes = comparison_data["batch_sizes"]
    
    # 绘制各个指标
    for metric_name, metric_config in METRICS_CONFIG.items():
        print(f"Plotting {metric_name}...")
        plot_metric(
            batch_sizes,
            comparison_data["metrics"][metric_name],
            metric_config,
            metric_name,
            output_dir
        )
    
    # 绘制加速比(基于 throughput)
    print("Plotting speedup comparison...")
    plot_speedup(
        batch_sizes,
        comparison_data["metrics"]["throughput"],
        args.baseline,
        output_dir
    )
    
    # 创建汇总表格
    print("Creating summary table...")
    create_summary_table(comparison_data, output_dir)
    
    print(f"\n{'='*80}")
    print(f"All plots saved to {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()