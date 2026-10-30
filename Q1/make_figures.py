"""步骤8：生成 Q1_v2 图表（全部从 results/ 程序化生成）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import dataio

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"

GRADE_COLOR = {"strong": "#2E7D32", "medium": "#F9A825", "weak": "#C62828"}
R = C.OUT_DIR


def save(fig, name):
    fig.savefig(C.FIG_DIR / name, dpi=200)
    plt.close(fig)
    print("saved", name)


def fig1():
    d = pd.read_csv(R / "paper_protocol.csv").sort_values("macro_test_r2")
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.macro_test_r2, height=0.4, label="逐维 macro R²", color="#1976D2")
    ax.barh(y + 0.2, d.var_weighted_test_r2, height=0.4, label="方差加权 R²",
            color="#90CAF9")
    ax.set_yticks(y)
    ax.set_yticklabels(d.model)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.21, color="#C62828", ls="--", lw=1)
    ax.text(0.211, len(d) - 0.6, "论文口径 0.21–0.22", color="#C62828", fontsize=9)
    for yi, v in zip(y - 0.2, d.macro_test_r2):
        ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7)
    lo = float(min(d.macro_test_r2.min(), d.var_weighted_test_r2.min())) - 0.03
    ax.set_xlim(lo, 0.32)
    ax.set_xlabel("R²（30 核心维，70/30 单切）")
    ax.set_title("图1  P-A 论文协议：候选模型对比")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig1_model_comparison.png")


def fig2():
    d = pd.read_csv(R / "per_target_ci.csv").sort_values("mean_r2")
    fig, ax = plt.subplots(figsize=(9, 12))
    y = np.arange(len(d))
    colors = [GRADE_COLOR[g] for g in d.grade]
    ax.errorbar(d.mean_r2, y, xerr=[d.mean_r2 - d["p2.5"], d["p97.5"] - d.mean_r2],
                fmt="none", ecolor="#555", elinewidth=0.8, capsize=2, zorder=1)
    ax.scatter(d.mean_r2, y, c=colors, s=28, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d.target, fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.10, color="#F9A825", ls=":", lw=1)
    ax.axvline(0.30, color="#2E7D32", ls=":", lw=1)
    ax.set_xlabel("ET 逐维 R²（30 折均值，误差线为 2.5%–97.5% 分位）")
    ax.set_title("图2  逐维 R² 与强度分级（绿=强，黄=中，红=弱）")
    for g, c in GRADE_COLOR.items():
        ax.scatter([], [], c=c, label=g)
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    save(fig, "fig2_per_target_r2.png")


def fig3():
    ab = pd.read_csv(R / "ablation_scale_views.csv")
    vs = pd.read_csv(R / "view_stability_summary.csv")
    views = ["V1_paper_all", "V2_scale_aware", "V3_absolute", "V4_relative", "V5_restored"]
    models = ["et", "pls", "ridge", "knn"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    for ax, kind in zip(axes[:2], ["core", "aux"]):
        sub = ab[ab.targets == kind]
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            d = sub[sub.model == m].set_index("view").reindex(views)
            ax.bar(np.arange(len(views)) + (i - 1.5) * w, d.macro_r2, width=w, label=m)
        ax.set_xticks(np.arange(len(views)))
        ax.set_xticklabels([v.split("_")[0] for v in views])
        ax.set_title(f"单切：{'核心' if kind=='core' else '辅助'}维 R²")
        ax.set_xlabel("尺度视图")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    ax = axes[2]
    d = vs.set_index("view").reindex(views)
    ax.bar(np.arange(len(views)), d.core_r2_mean, yerr=d.core_r2_sd,
           color=["#B0BEC5", "#64B5F6", "#1976D2", "#EF5350", "#90A4AE"],
           capsize=3)
    ax.set_xticks(np.arange(len(views)))
    ax.set_xticklabels([v.split("_")[0] for v in views])
    ax.set_title("30 折：固定 ET 参数")
    ax.set_xlabel("尺度视图")
    ax.set_ylim(0, 0.25)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("图3  尺度视图消融（V3 仅绝对 / V4 仅相对）", y=1.02)
    save(fig, "fig3_scale_ablation.png")


def fig4():
    ld = pd.read_csv(R / "level_distribution.csv")
    ld = ld[ld.scheme == "quantile"]
    ci = pd.read_csv(R / "per_target_ci.csv")[["target", "mean_r2"]]
    piv = ld.pivot(index="target", columns="level", values="n").reindex(
        ci.sort_values("mean_r2", ascending=False).target)
    share = piv.div(piv.sum(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 11),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    im = ax.imshow(share, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.6)
    ax.set_xticks(range(7))
    ax.set_xticklabels(range(7))
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels(share.index, fontsize=7)
    ax.set_xlabel("相对等级（0=最低 1/7，6=最高 1/7）")
    ax.set_title("逐维等级分布（行归一化，quantile 方案）")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax2 = axes[1]
    d = pd.read_csv(R / "per_target_calibration.csv")
    bp = ax2.boxplot([d.mid_share_raw, d.mid_share_cal], labels=["原始", "校准"],
                     patch_artist=True, widths=0.5)
    for p, c in zip(bp["boxes"], ["#90CAF9", "#1976D2"]):
        p.set_facecolor(c)
    ax2.set_ylabel("3–4 档占比")
    ax2.set_title("等级堆积")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("图4  相对 0–6 等级：分布与校准效果", y=0.995)
    save(fig, "fig4_level_distribution.png")


def fig5():
    d = pd.read_csv(R / "cross_source_summary.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.scatter(d.spearman_obs, d.spearman_pred, s=30, c="#1976D2", alpha=0.75)
    lim = [-0.2, 0.8]
    ax.plot(lim, lim, ls="--", color="gray", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="k", lw=0.6)
    for _, r in d.iterrows():
        if r.spearman_obs > 0.55 or r.spearman_pred > 0.65 or r.spearman_obs < 0.02 \
                or r.spearman_pred < -0.05:
            ax.annotate(r.target, (r.spearman_obs, r.spearman_pred), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("观测感官分 vs 评论指标 Spearman")
    ax.set_ylabel("模型预测 vs 评论指标 Spearman")
    ax.set_title("图5  跨源一致性（虚线为 y=x）")
    ax.grid(alpha=0.3)
    save(fig, "fig5_cross_source.png")


def fig6():
    b = dataio.load_all()
    Y = b.Y("model")
    oof = pd.read_csv(R / "oof_et_20261023.csv", index_col=0).loc[b.beer, Y.columns]
    picks = ["Acidity", "Hops aroma - overall", "Esters taste - floral"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, t in zip(axes, picks):
        x, y = oof[t].to_numpy(), Y[t].to_numpy()
        ax.scatter(x, y, s=12, alpha=0.6, c="#1976D2")
        rho = spearmanr(x, y).statistic
        k, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, k * xs + b0, color="#C62828", lw=1)
        ax.set_title(f"{t}\nSpearman={rho:.3f}", fontsize=9)
        ax.set_xlabel("折外预测")
        ax.set_ylabel("观测")
        ax.grid(alpha=0.3)
    fig.suptitle("图6  折外预测 vs 观测（强/中/弱各一例）", y=1.03)
    save(fig, "fig6_oof_pred_vs_obs.png")


def fig7():
    d = pd.read_csv(R / "stability_folds.csv")
    d = d[d.model.isin(["et", "rf"])]
    ps = d.groupby(["seed", "model"]).core_r2.mean().unstack()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, c in [("et", "#1976D2"), ("rf", "#EF6C00")]:
        ax.plot(range(len(ps)), ps[m], marker="o", label=m.upper(), color=c)
    ax.set_xticks(range(len(ps)))
    ax.set_xticklabels([str(s)[-2:] for s in ps.index])
    ax.set_xlabel("划分 seed（后两位）")
    ax.set_ylabel("core macro R²（3 折均值）")
    ax.set_title("图7  ET 与 RF 的逐 seed 稳定性")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "fig7_et_rf_stability.png")


def main():
    C.ensure_dirs()
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()


if __name__ == "__main__":
    main()
