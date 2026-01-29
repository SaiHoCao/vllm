#!/usr/bin/env python3
"""
科学验证不同 replay_mode 下的输出一致性
"""
import json
import hashlib
import os
from typing import List, Dict, Any
import random
from vllm import LLM, SamplingParams


def save_outputs(outputs, mode_name: str, save_dir: str = "./mode_outputs") -> List[Dict]:
    """保存输出用于对比"""
    os.makedirs(save_dir, exist_ok=True)
    
    results = []
    for i, output in enumerate(outputs):
        results.append({
            "id": i,
            "prompt_tokens": len(output.prompt_token_ids),
            "output_token_ids": output.outputs[0].token_ids,
            "output_text": output.outputs[0].text,
        })
    
    filepath = os.path.join(save_dir, f"{mode_name}.json")
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved {mode_name}: {len(results)} outputs to {filepath}")
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


def run_mode_comparison(model_path: str, prompts: List[str]):
    """运行所有模式并比较"""
    modes = ["DUAL_PARALLEL", "DUAL_SERIAL", "PADDING"]
    
    # 确定性采样
    sampling_params = SamplingParams(
        max_tokens=64,
        temperature=0.0,
        ignore_eos=True,
        seed=42,
    )
    
    all_results = {}
    
    for mode in modes:
        print(f"\n{'#'*70}")
        print(f"# Running mode: {mode}")
        print(f"{'#'*70}")

            
        
        llm = LLM(
            model=model_path,
            max_model_len=8192,
            compilation_config={
                "level": "3",
                "cudagraph_mode": "FULL_DECODE_ONLY",
                "replay_mode": mode,
                "cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,192,256],
            },
            enable_chunked_prefill=False,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        results = save_outputs(outputs, mode)
        all_results[mode] = results
        
        del llm
        import torch
        torch.cuda.empty_cache()
    
    # 交叉比较
    print(f"\n{'#'*70}")
    print(f"# Cross-Mode Validation")
    print(f"{'#'*70}")
    
    baseline = modes[0]
    all_pass = True
    
    for mode in modes[1:]:
        passed = compare_outputs(
            all_results[baseline], 
            all_results[mode],
            baseline, 
            mode
        )
        all_pass = all_pass and passed
    
    print(f"\n{'='*70}")
    if all_pass:
        print("🎉 ALL MODES PRODUCE IDENTICAL OUTPUTS!")
    else:
        print("⚠️  SOME MODES PRODUCE DIFFERENT OUTPUTS!")
    print(f"{'='*70}")


if __name__ == "__main__":
    # 准备测试数据
    prompts = [
        "The quick brown fox jumps over the lazy dog. " * 20 + "\n\nSummarize:",
        "Python is a programming language. " * 15 + "\n\nWhat is this about?",
    ] * 5  # 10 个 prompts
    
    run_mode_comparison(
        model_path="/home/csh/data/Qwen3-4B",
        prompts=prompts
    )
