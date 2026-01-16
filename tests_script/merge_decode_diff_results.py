"""
合并和分析 Benchmark 测试结果
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def load_results(input_dir: Path):
    """加载所有配置的测试结果"""
    
    results = {}
    
    # 查找所有结果文件
    for result_file in input_dir.glob("*_result.json"):
        config_name = result_file.stem.replace("_result", "")
        
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[config_name] = data
                print(f"✅ Loaded {config_name} from {result_file}")
        except Exception as e:
            print(f"❌ Failed to load {result_file}: {e}")
    
    return results


def merge_results(loaded_results: dict):
    """合并所有配置的测试结果"""
    
    merged = {
        "timestamp": datetime.now().isoformat(),
        "configs": {},
        "batch_sizes": set()
    }
    
    for config_name, result in loaded_results.items():
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
                "tpot": r.get("tpot", 0) * 1000,  # 转换为 ms
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


def print_summary(merged_results: dict):
    """打印结果摘要"""
    
    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    
    for config_name, config_data in merged_results["configs"].items():
        print(f"\n{config_name}:")
        print(f"  Description: {config_data['description']}")
        print(f"  Status: {'✅ Success' if config_data['success'] else '❌ Failed'}")
        print(f"  Time: {config_data['elapsed_time']:.1f}s")
        print(f"  Results: {len(config_data['results'])} batch sizes")


def main():
    parser = argparse.ArgumentParser(description="Merge and analyze benchmark results")
    parser.add_argument("--input-dir", type=str, default="2_benchmark_decode_results",
                       help="Input directory with results")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(exist_ok=True)
    
    # 加载结果
    print(f"\n{'='*80}")
    print("Loading Results")
    print(f"{'='*80}")
    
    loaded_results = load_results(input_dir)
    
    if not loaded_results:
        print("❌ No results found!")
        return
    
    # 合并结果
    print(f"\n{'='*80}")
    print("Merging Results")
    print(f"{'='*80}")
    
    merged_results = merge_results(loaded_results)
    comparison_data = create_comparison_data(merged_results)
    
    # 保存合并结果
    merged_file = output_dir / "merged_results.json"
    with open(merged_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    print(f"✅ Merged results saved to {merged_file}")
    
    # 保存比较数据
    comparison_file = output_dir / "comparison_data.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✅ Comparison data saved to {comparison_file}")
    
    # 打印摘要
    print_summary(merged_results)
    
    print(f"\n{'='*80}")
    print(f"All merged data saved to {output_dir}")
    print(f"Use comparison_data.json for plotting")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()