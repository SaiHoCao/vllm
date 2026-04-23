import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def load_data(base_dir=".") -> pd.DataFrame:
    """读取所有符合规则的 JSON 文件并合并为 DataFrame"""
    modes = ["inplace", "dual", "pad"]
    seq_lens = [1024, 2048]
    model_size = "4b"
    
    all_data = []
    
    for mode in modes:
        for seq in seq_lens:
            file_name = f"decode_benchmark_results_{mode}_{model_size}_{seq}_256.json"
            file_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(file_path):
                print(f"Loading {file_name}...")
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for row in data:
                        if row.get('success', False):
                            row['mode'] = mode
                            row['tpot_ms'] = row['tpot'] * 1000 
                            row['file_seq_len'] = seq 
                            # 确保 padding_batch 存在
                            if 'padding_batch' not in row:
                                row['padding_batch'] = row.get('matched_bucket', row['batch_size']) - row['batch_size']
                            all_data.append(row)
            else:
                print(f"Warning: File not found: {file_path}")
                
    if not all_data:
         raise ValueError("No data loaded. Please check your file paths and names.")
            
    df = pd.DataFrame(all_data)
    df = df.sort_values(by='batch_size')
    return df

def plot_benchmark_results(df: pd.DataFrame):
    """绘制性能测试结果的 2x2 子图"""
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle('vLLM Decode Benchmark: Modes vs Sequence Lengths', fontsize=18, fontweight='bold', y=0.98)

    metrics = [
        {"col": "tpot_ms", "title": "TPOT (Time Per Output Token)", "ylabel": "TPOT (ms)"},
        {"col": "pure_decode_time", "title": "Pure Decode Time", "ylabel": "Time (seconds)"}
    ]
    seq_lens = sorted(df['file_seq_len'].unique())

    # 提取 Padding 信息，增加 padding_batch 字段
    padding_df = df[['batch_size', 'matched_bucket', 'padding_waste_pct', 'padding_batch']].drop_duplicates(subset=['batch_size']).sort_values('batch_size')

    for row_idx, metric in enumerate(metrics):
        for col_idx, seq in enumerate(seq_lens):
            ax = axes[row_idx, col_idx]
            
            subset = df[df['file_seq_len'] == seq]
            
            # 1. 绘制主要指标折线图
            sns.lineplot(
                data=subset, 
                x='batch_size', 
                y=metric['col'], 
                hue='mode', 
                style='mode', 
                markers=True, 
                dashes=False, 
                linewidth=2.5,
                markersize=8,
                ax=ax,
                palette="Set1"
            )
            
            ax.set_title(f"{metric['title']} (Seq Len = {seq})", fontsize=14)
            ax.set_ylabel(metric['ylabel'], fontsize=12, fontweight='bold')
            ax.set_xlabel("Batch Size" if row_idx == 1 else "", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(title='Mode', loc='upper left')
            else:
                ax.get_legend().remove()

            # 2. 绘制 Padding Waste 比例 (右侧 Y 轴)
            ax2 = ax.twinx()
            
            # 根据你的最小 batch size 间距动态调整柱子宽度
            min_gap = padding_df['batch_size'].diff().min()
            bar_width = min_gap * 0.4 if pd.notna(min_gap) else 15
            
            ax2.bar(
                padding_df['batch_size'], 
                padding_df['padding_waste_pct'], 
                width=bar_width, 
                color='gray', 
                alpha=0.2, 
                label='Padding Waste %'
            )
            
            # 3. 在柱状图上方标注绝对的 Pad Batch 数量
            for _, row_data in padding_df.iterrows():
                pad_amount = int(row_data['padding_batch'])
                pct = row_data['padding_waste_pct']
                x_pos = row_data['batch_size']
                
                if pad_amount > 0:
                    ax2.text(
                        x_pos,
                        pct + 1,  # 在柱子顶端上方稍微偏移一点
                        f"+{pad_amount}", 
                        color='dimgray', 
                        fontsize=9, 
                        fontweight='bold',
                        ha='center', 
                        va='bottom',
                        rotation=90 if bar_width < 20 else 0  # 如果柱子太密，文字竖排防重叠
                    )

            ax2.set_ylabel("Padding Waste %", color='gray', fontsize=10)
            ax2.tick_params(axis='y', labelcolor='gray')
            ax2.yaxis.set_major_formatter(PercentFormatter())
            # 设置 ylim 留出更多空间给文字标签
            max_pct = padding_df['padding_waste_pct'].max()
            ax2.set_ylim(0, max(max_pct * 1.5, 100)) 
            ax2.grid(False) 

            # 4. 标注刚好无 Padding 的 Bucket 锚点
            zero_pad_points = padding_df[padding_df['padding_waste_pct'] == 0]
            for _, point in zero_pad_points.iterrows():
                bucket_val = int(point['batch_size'])
                ax.axvline(x=bucket_val, color='green', linestyle=':', alpha=0.5)
                ax.text(
                    bucket_val, ax.get_ylim()[0], f" Bkt:\n{bucket_val}", 
                    color='green', fontsize=8, verticalalignment='bottom'
                )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_png = "256_vllm_benchmark_analysis_with_pad_amounts.png"
    plt.savefig(output_png, dpi=300)
    print(f"\n✅ Plot saved successfully to: {output_png}")
    plt.show()

if __name__ == "__main__":
    df = load_data()
    print(f"Successfully loaded {len(df)} data points.")
    plot_benchmark_results(df)