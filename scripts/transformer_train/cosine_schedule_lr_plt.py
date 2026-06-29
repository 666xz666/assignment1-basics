import os
import matplotlib.pyplot as plt
import numpy as np

# 你的参考学习率序列
expected_lrs = [
    0,
    0.14285714285714285,
    0.2857142857142857,
    0.42857142857142855,
    0.5714285714285714,
    0.7142857142857143,
    0.8571428571428571,
    1.0,
    0.9887175604818206,
    0.9554359905560885,
    0.9018241671106134,
    0.8305704108364301,
    0.7452476826029011,
    0.6501344202803414,
    0.55,
    0.44986557971965857,
    0.3547523173970989,
    0.26942958916356996,
    0.19817583288938662,
    0.14456400944391146,
    0.11128243951817937,
    0.1,
    0.1,
    0.1,
    0.1,
]

# 构造迭代步数 t
t_list = np.arange(len(expected_lrs))

# 该组调度超参
Tw = 7
Tc = 20
alpha_max = 1.0
alpha_min = 0.1

# 创建画布
plt.figure(figsize=(12, 6), dpi=120)
plt.plot(
    t_list,
    expected_lrs,
    color="#2563eb",
    linewidth=2.2,
    marker="o",
    markersize=3.5,
    label=r"$\alpha_t$ Expected Learning Rate",
)

# 划分三个区间竖线+标注
plt.axvline(
    x=Tw, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Warmup End $T_w={Tw}$"
)
plt.axvline(
    x=Tc,
    color="#22a75d",
    linestyle="--",
    linewidth=1.5,
    label=f"Annealing End $T_c={Tc}$",
)

# 区间文字标注
plt.text(
    x=Tw / 2,
    y=alpha_max * 0.92,
    s="Warm-up\nLinear Rise",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", alpha=0.7),
)
plt.text(
    x=(Tw + Tc) / 2,
    y=alpha_max * 0.92,
    s="Cosine Annealing\nDecay",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0fdf4", alpha=0.7),
)
plt.text(
    x=(Tc + len(t_list) - 1) / 2,
    y=alpha_max * 0.92,
    s="Post-Annealing\nConstant LR",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="#eff6ff", alpha=0.7),
)

# 坐标轴与标题
plt.xlabel("Iteration $t$", fontsize=11)
plt.ylabel(r"Learning Rate $\alpha_t$", fontsize=11)
plt.title("Warmup + Cosine Annealing Learning Rate Schedule", fontsize=13, pad=12)
plt.grid(True, alpha=0.3, linestyle="-")
plt.legend(fontsize=9.5)
plt.tight_layout()

# 目标保存路径
save_dir = "./output/logs/plt"
os.makedirs(save_dir, exist_ok=True)  # 自动递归创建文件夹，不存在则新建
save_path = os.path.join(save_dir, "lr_cosine_schedule_curve.png")

# 保存图片，不调用plt.show()
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()  # 释放画布内存，避免内存堆积
print(f"曲线已保存至: {save_path}")
