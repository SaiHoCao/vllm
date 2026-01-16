import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
input_sizes = ['150', '200', '250', '300', '350', '400', '450']

# 数据来自之前的表格
data = {
    "Initial Mem": {
        "Pad":  [130238] * 7,
        "Dual": [130250] * 7
    },
    "Decode Initial": {
        "Pad":  [128178, 128220, 128266, 128308, 128352, 128394, 128440],
        "Dual": [128192, 128236, 128298, 128340, 128384, 128426, 128472]
    },
    "Before Exe Mem": {
        "Pad":  [128046] * 7,
        "Dual": [128058] * 7
    },
    "Execution Mem": {
        "Pad":  [128046] * 7,
        "Dual": [128060, 128060, 128078, 128078, 128078, 128078, 128078]
    }
}

# 2. 设置绘图参数
x = np.arange(len(input_sizes))  # X轴刻度
width = 0.35  # 柱子宽度

# 创建 2x2 的子图布局
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pad Mode vs Dual Mode Memory Comparison (MB)', fontsize=16)

# 扁平化数组以便遍历
axes_flat = axs.flatten()
metrics = list(data.keys())

# 3. 循环绘制每个指标
for i, metric in enumerate(metrics):
    ax = axes_flat[i]
    pad_vals = data[metric]["Pad"]
    dual_vals = data[metric]["Dual"]
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, pad_vals, width, label='Pad Mode', color='#4c72b0')
    rects2 = ax.bar(x + width/2, dual_vals, width, label='Dual Mode', color='#dd8452')
    
    # 设置标题和标签
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(input_sizes)
    ax.set_xlabel('Input Batch Size')
    ax.set_ylabel('Memory (MB)')
    ax.legend()
    
    # *** 关键步骤：动态调整 Y 轴范围 ***
    # 因为数值很大但差异很小，我们需要缩放 Y 轴只显示顶部
    all_vals = pad_vals + dual_vals
    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) if (max_val - min_val) > 0 else 10
    # 上下各留一点余地，如果 margin 为 0 (数值完全一样) 则默认留 20
    if margin == 0: margin = 20
    
    ax.set_ylim(min_val - margin*1.5, max_val + margin*1.5)
    
    # 在柱子上标注差异值 (Dual - Pad)
    # 为了保持整洁，我们只标注 Dual 比 Pad 高出的部分
    for j in range(len(x)):
        diff = dual_vals[j] - pad_vals[j]
        # 在 Dual 柱子上方标注差值
        ax.text(x[j] + width/2, dual_vals[j] + margin*0.1, f'+{diff}', 
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局防止重叠
# plt.show()
plt.savefig('memory_comparison_ttsh.png', dpi=300)
