"""Q3 图表：从 results/ 的最终结果重建论文用图。

输出：
  figures/fig_q3_hit.png
  figures/fig_q3_fp.png
  figures/fig_q3_sensory.png
  figures/fig_q3_transfer.png
  figures/fig_q3_defect.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import q3_config as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11


def save(fig, name):
    fig.savefig(C.FIG_DIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def hit():
    d = pd.read_csv(C.OUT_DIR / "metrics_by_style.csv", encoding="utf-8-sig")
    d = d[d.low_confidence == False].sort_values("style_hit_rate")
    mean = float(d.style_hit_rate.mean())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["style"], d.style_hit_rate, color="#2E75B6")
    ax.axvline(mean, color="#C55A11", ls="--", lw=1.5,
               label=f"{len(d)} 个风格均值 {mean:.3f}")
    ax.set_xlabel("风格命中率")
    ax.set_xlim(0, 1.02)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=False)
    save(fig, "fig_q3_hit.png")


def fingerprint():
    d = pd.read_csv(C.OUT_DIR / "fingerprint_check.csv", encoding="utf-8-sig")
    d = d.sort_values("mean_in_interval")
    mean = float(d.mean_in_interval.mean())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(d["style"], d.mean_in_interval, color="#2E75B6")
    ax.axvline(mean, color="#C55A11", ls="--", lw=1.5,
               label=f"{len(d)} 个风格均值 {mean:.3f}")
    ax.set_xlabel("指纹区间覆盖率")
    ax.set_xlim(0.6, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), frameon=False)
    save(fig, "fig_q3_fp.png")


def sensory():
    gen = pd.read_csv(C.OUT_DIR / "sensory_profiles_generated.csv")
    folds = pd.read_csv(C.OUT_DIR / "folds.csv")
    sens = pd.read_csv(C.DATA_DIR / "4_感官分析结果.csv",
                       encoding="utf-8-sig").set_index("beer")
    tr = folds[folds.split == "outer_train"]["beer"]
    real = sens.loc[tr]
    dims = [("Acidity", "酸度"), ("Bitterness", "苦味"), ("Sweetness", "甜度"),
            ("Alcohol", "酒精感"), ("Body", "醇厚度"),
            ("Esters taste - overall", "酯类风味")]
    styles = ["Tripel", "Kriek", "Lambic"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, s in zip(axes, styles):
        g = gen[gen["style"] == s][[c for c, _ in dims]].mean()
        r = real[real["tasting_category_fine"] == s][[c for c, _ in dims]].mean()
        x = np.arange(len(dims))
        ax.plot(x, g.to_numpy(), "o-", color="#2E75B6", lw=2, label="生成谱预测均值")
        ax.plot(x, r.to_numpy(), "s--", color="#7F7F7F", lw=2, label="真实训练样本均值")
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in dims], rotation=45, ha="right")
        ax.set_title(s, fontsize=14)
        ax.axhline(0, color="k", lw=0.6)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("感官均值（连续变换尺度）")
    axes[0].legend(loc="best", frameon=False, fontsize=10)
    save(fig, "fig_q3_sensory.png")


def transfer():
    d = pd.read_csv(C.OUT_DIR / "transfer_metrics.csv", encoding="utf-8-sig")
    cases = [("Pils/Lager->Tripel", "Pils/Lager→Tripel"),
             ("Blond->Tripel", "Blond→Tripel"),
             ("Kriek->Lambic", "Kriek→Lambic")]
    methods = ["T0", "T1", "T3"]
    colors = ["#7F7F7F", "#9DC3E6", "#2E75B6", "#C55A11"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.19
    x = np.arange(len(cases))
    for i, (case, _) in enumerate(cases):
        sub = d[d.case == case].set_index("method")
        vals = [float(d[(d.case == case)].p_target_source.iloc[0])]
        for m in methods:
            vals.append(float(sub.loc[m, "p_target_trans"]))
        pos = x[i] + (np.arange(4) - 1.5) * width
        ax.bar(pos, vals, width=width, color=colors)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cases])
    ax.set_ylabel("目标风格得分")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(
        colors, ["迁移前", "T0 均值平移", "T1 协方差对齐", "T3 源值约束"])]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_q3_transfer.png")


def defect():
    d = pd.read_csv(C.OUT_DIR / "transfer_metrics.csv", encoding="utf-8-sig")
    cases = [("Pils/Lager->Tripel", "Pils/Lager→Tripel"),
             ("Blond->Tripel", "Blond→Tripel"),
             ("Kriek->Lambic", "Kriek→Lambic")]
    methods = ["T0", "T1", "T3"]
    colors = ["#7F7F7F", "#9DC3E6", "#2E75B6", "#C55A11"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.19
    x = np.arange(len(cases))
    for i, (case, _) in enumerate(cases):
        sub = d[d.case == case].set_index("method")
        vals = [float(d[(d.case == case)].dms_oav_source.iloc[0])]
        vals += [float(sub.loc[m, "dms_oav_trans"]) for m in methods]
        pos = x[i] + (np.arange(4) - 1.5) * width
        ax.bar(pos, vals, width=width, color=colors)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cases])
    ax.set_ylabel("DMS 的 OAV 中位数")
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for c, l in zip(
        colors, ["迁移前", "T0 均值平移", "T1 协方差对齐", "T3 源值约束"])]
    ax.legend(handles=handles, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_q3_defect.png")


def main():
    C.ensure_dirs()
    hit()
    fingerprint()
    sensory()
    transfer()
    defect()


if __name__ == "__main__":
    main()
