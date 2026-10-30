# -*- coding: utf-8 -*-
"""图 5-1：上侧远超单元在化学指标和样本中的分布（读清洗后审计结果，可复现）。"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 9

HERE = Path(__file__).resolve().parent
BASE = str(HERE / "清洗后")
OUT_PDF = str(HERE / "清洗后" / "outlier_distribution.pdf")
OUT_PNG = str(HERE / "清洗后" / "fig_outlier.png")

pc = pd.read_csv(os.path.join(BASE, "异常值审计_逐列.csv"), encoding="utf-8-sig")
bb = pd.read_csv(os.path.join(BASE, "异常值_按beer汇总.csv"), encoding="utf-8-sig")

col_counts = pc["n_Tukey远超"].sort_values(ascending=False)
col_counts = col_counts[col_counts > 0].to_numpy()
beer_counts = bb["远超单元数"].sort_values(ascending=False)
beer_counts = beer_counts[beer_counts > 0].to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), dpi=300)
fig.suptitle("上侧远超单元在化学指标和样本中的分布", fontsize=10)

ax = axes[0]
ax.bar(range(len(col_counts)), col_counts, color="#4472C4", width=0.8)
ax.set_title("(a) 各化学指标的上侧远超单元数")
ax.set_xlabel("化学指标（按计数降序）")
ax.set_ylabel("远超单元数")
ax.grid(axis="y", color="#E6E6E6", zorder=0)
ax.set_axisbelow(True)

ax = axes[1]
ax.bar(range(len(beer_counts)), beer_counts, color="#ED7D31", width=0.8,
       edgecolor="#C55A11", linewidth=0.2)
ax.set_title("(b) 各样本的上侧远超单元数")
ax.set_xlabel("样本（按计数降序）")
ax.set_ylabel("远超单元数")
ax.grid(axis="y", color="#E6E6E6", zorder=0)
ax.set_axisbelow(True)

for ax in axes:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
print("cols", len(col_counts), "units", int(col_counts.sum()),
      "| beers", len(beer_counts), "units", int(beer_counts.sum()))

