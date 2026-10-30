"""Q2 Step 9：图表（全部从 results/ 程序化生成）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import q2_config as C
import graph as G

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"
OUT = C.OUT_DIR


def save(fig, name):
    fig.savefig(C.FIG_DIR / name, dpi=250)
    plt.close(fig)
    print("saved", name)


def network():
    m2 = pd.read_csv(C.DATA_DIR / "2_化学参数Spearman相关矩阵.csv",
                     index_col=0, encoding="utf-8-sig")
    thr = G.top_quantile_threshold(m2, C.EDGE_QUANTILE)
    Gp, _, _ = G.build_graph(m2, thr)
    _, labels = G.communities(Gp)
    nodes = G.node_table(Gp, labels)
    pos = nx.spring_layout(Gp, seed=0, k=0.55, iterations=150)
    comms = sorted(set(labels.values()))
    cmap = plt.get_cmap("tab20")
    color = {c: cmap(i % 20) for i, c in enumerate(comms)}
    colors = [color[labels[n]] for n in Gp.nodes]
    sizes = [15 + 7 * nodes.set_index("compound").loc[n, "weighted_degree"]
             for n in Gp.nodes]
    fig, ax = plt.subplots(figsize=(14, 12))
    nx.draw_networkx_edges(Gp, pos, ax=ax, alpha=0.10, width=0.5, edge_color="#777")
    nx.draw_networkx_nodes(Gp, pos, ax=ax, node_color=colors, node_size=sizes,
                           linewidths=0.4, edgecolors="white")
    # 派生节点黑边
    derived = [n for n in Gp.nodes if n in C.DERIVED_COLS]
    nx.draw_networkx_nodes(Gp, pos, nodelist=derived, ax=ax, node_color="none",
                           node_size=[15 + 7 * nodes.set_index("compound").loc[n, "weighted_degree"]
                                      for n in derived],
                           edgecolors="black", linewidths=1.2)
    top = nodes.sort_values("weighted_degree", ascending=False).head(10)
    for _, r in top.iterrows():
        x, y = pos[r.compound]
        ax.text(x, y, r.compound, fontsize=7, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.6))
    ax.set_title(f"Q2 共表达图（top 5% |ρ|；{Gp.number_of_nodes()} 化合物，"
                 f"{Gp.number_of_edges()} 正边，{len(comms)} 社区；灰黑边=派生列）")
    ax.axis("off")
    save(fig, "q2_network.png")


def top3_heatmap():
    d = pd.read_csv(OUT / "final" / "q2_top3_final.csv")
    d = d[d.view == "V3"]
    summ = pd.read_csv(OUT / "final" / "q2_stability_final.csv")
    summ = summ[summ.view == "V3"].sort_values(["n_stable", "target"],
                                               ascending=[False, True])
    order = summ.target.tolist()
    pivot = d.pivot(index="target", columns="rank", values="rho_adjusted").reindex(order)
    comp = d.pivot(index="target", columns="rank", values="compound").reindex(order)
    status = d.pivot(index="target", columns="rank", values="status_final").reindex(order)
    sym = {"stable": "●", "exploratory": "○", "definitional": "◆", "not_supported": "×"}
    half = (len(order) + 1) // 2
    parts = [order[:half], order[half:]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 8),
                             gridspec_kw={"wspace": 0.65})
    im = None
    for k, (ax, part) in enumerate(zip(axes, parts)):
        p = pivot.loc[part]
        c = comp.loc[part]
        s = status.loc[part]
        im = ax.imshow(p.to_numpy(), aspect="auto", cmap="coolwarm",
                       vmin=-0.6, vmax=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Top-1", "Top-2", "Top-3"], fontsize=10)
        ax.set_yticks(range(len(part)))
        ax.set_yticklabels([str(t).replace("Overall appreciation (Trained panel)",
                                           "Overall appreciation") for t in part],
                           fontsize=9)
        for i, t in enumerate(part):
            for j, rank in enumerate([1, 2, 3]):
                name = c.loc[t, rank]
                if pd.isna(name):
                    continue
                ax.text(j, i, f"{sym.get(s.loc[t, rank], '')} {str(name)[:15]}",
                        ha="center", va="center", fontsize=7.2, color="black")
        ax.set_title("前 25 个维度" if k == 0 else "后 25 个维度", fontsize=11)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("风格调整后偏相关 ρ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle("各感官维度前三候选的风格偏相关（V3；● stable / ○ exploratory / ◆ definitional；颜色=adjusted ρ）",
                 fontsize=13, y=1.02)
    save(fig, "q2_top3_heatmap.png")


def stability():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for v, key in [("V3", "v3"), ("V2", "v2")]:
        d = pd.read_csv(OUT / "stability" / f"q2_stability_selection_{key}.csv")
        per = d.groupby("target").boot_freq.max()
        axes[0].boxplot(per, positions=[1 if v == "V3" else 2], widths=0.5,
                        patch_artist=True,
                        boxprops=dict(facecolor="#64B5F6" if v == "V3" else "#90CAF9"))
    axes[0].set_xticks([1, 2]); axes[0].set_xticklabels(["V3", "V2"])
    axes[0].set_ylabel("每目标最大 bootstrap 频率")
    axes[0].set_title("稳定性选择")
    axes[0].grid(axis="y", alpha=0.3)

    j = pd.read_csv(OUT / "consensus" / "q2_method_jaccard.csv")
    j = j[j.k == 3].groupby(["view", "method_a", "method_b"]).jaccard.mean().reset_index()
    labels = [f"{r.view}\n{r.method_a[:4]}-{r.method_b[:4]}" for r in j.itertuples()]
    axes[1].bar(range(len(j)), j.jaccard, color="#1976D2")
    axes[1].set_xticks(range(len(j))); axes[1].set_xticklabels(labels, fontsize=6)
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Top-3 Jaccard")
    axes[1].set_title("多方法一致性")
    axes[1].grid(axis="y", alpha=0.3)

    vj = pd.read_csv(OUT / "consensus" / "q2_view_jaccard.csv")
    axes[2].hist(vj.jaccard_top3, bins=np.linspace(0, 1, 11), color="#EF6C00", alpha=0.8)
    axes[2].set_xlabel("V3 vs V2 Top-3 Jaccard")
    axes[2].set_ylabel("目标数")
    axes[2].set_title(f"视图一致性（中位 {vj.jaccard_top3.median():.2f}）")
    axes[2].grid(axis="y", alpha=0.3)
    fig.suptitle("Q2 稳定性与一致性", y=1.02)
    save(fig, "q2_stability.png")


def style_scatter():
    d = pd.read_csv(OUT / "style" / "q2_style_sensitivity.csv")
    d = d[d.view == "V3"]
    color = {"style_conditional": "#1976D2", "style_proxy": "#C62828",
             "uncertain": "#F9A825", "definitional": "#6A1B9A"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for flag, g in d.groupby("style_flag"):
        ax.scatter(g.rho_raw, g.rho_adjusted, s=22, alpha=0.75,
                   label=flag, c=color.get(flag, "gray"))
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    # 坐标范围由数据决定，保证所有点（含 |ρ|>0.6 与 ρ<−0.2）完整可见
    lo = min(d.rho_raw.min(), d.rho_adjusted.min()) - 0.04
    hi = max(d.rho_raw.max(), d.rho_adjusted.max()) + 0.04
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    # 标注正文引用的“目标—化合物”对（避免同一化合物在别的维度被误标）
    named = [("Hops aroma - overall", "Iron"), ("Dimethyl sulfide", "Protein"),
             ("Acidity", "Lactic acid"), ("Acidity", "pH"), ("Acidity", "Acetic acid"),
             ("Body", "Ethanol"), ("Malt aroma - overall", "Color"),
             ("Esters aroma - overall", "Ethyl acetate")]
    for k, (tg, cp) in enumerate(named):
        r = d[(d.target == tg) & (d.compound == cp)]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.annotate(cp, (r.rho_raw, r.rho_adjusted), fontsize=6,
                    xytext=(4, 4 if k % 2 == 0 else -9), textcoords="offset points")
    ax.set_xlabel("原始 Spearman ρ"); ax.set_ylabel("风格调整后 ρ")
    ax.set_title(f"Q2 风格偏相关：raw vs adjusted（V3，n={len(d)}）")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    save(fig, "q2_style.png")


def drop_scatter():
    d = pd.read_csv(OUT / "dropcolumn" / "q2_dropcolumn.csv")
    d = d[(d.view == "V3") & (d.drop_type == "group")]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0.02, color="#F9A825", ls="--", lw=1)
    ax.axvline(0.01, color="#2E7D32", ls="--", lw=1)
    ax.scatter(d.delta_r2, d.delta_rmse_pct, s=25, c="#1976D2", alpha=0.75)
    for _, r in d.iterrows():
        if r.target in ["Acidity", "Hops aroma - overall", "Esters aroma - floral",
                        "Dimethyl sulfide"]:
            ax.annotate(r.target, (r.delta_r2, r.delta_rmse_pct), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("整组 ΔR²"); ax.set_ylabel("整组 ΔRMSE%")
    ax.set_title("Q2 drop-column（V3；3 折 OOF）")
    ax.grid(alpha=0.3)
    save(fig, "q2_dropcolumn.png")


def status_counts():
    d = pd.read_csv(OUT / "final" / "q2_top3_final.csv")
    tab = d.groupby(["view", "status_final"]).size().unstack(fill_value=0)
    order = ["stable", "exploratory", "definitional", "not_supported"]
    tab = tab.reindex(columns=[c for c in order if c in tab.columns], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(tab))
    colors = {"stable": "#2E7D32", "exploratory": "#F9A825",
              "definitional": "#6A1B9A", "not_supported": "#C62828"}
    for c in tab.columns:
        ax.bar(tab.index, tab[c], bottom=bottom, label=c, color=colors[c])
        bottom += tab[c].to_numpy()
    ax.set_ylabel("Top-3 行数")
    ax.set_title("Q2 最终状态（150 行/视图）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "q2_status_counts.png")


def main():
    C.ensure_dirs()
    network()
    top3_heatmap()
    stability()
    style_scatter()
    drop_scatter()
    status_counts()


if __name__ == "__main__":
    main()
