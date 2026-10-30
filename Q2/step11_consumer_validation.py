"""Q2 Step 11: 用附件 8.xlsx 的消费者喜好输出做外部锚点对照。

数据：附件/8.xlsx 的 `GBM SHAP Values`。SHAP 可加：
    consumer_pred = base_prediction + Σ SHAP
得到 250 款啤酒的消费者喜好预测。它是模型输出，不是原始评分；只作
方向性外部锚点，不做独立验证。再做三层对照：
  1) 目标层：consumer_pred 与品评组 overall 观测/预测、价格
  2) 化合物层：consumer_pred 的单变量 Spearman vs 我们的 stable/重要性
  3) 类别层：consumer 重要性与我们的化学类别组级 ΔR² 对照

输出：results/consumer/
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.bbox"] = "tight"

import q2_config as C
import dataio
from step10_group_importance import chem_class

P8 = C.ROOT / "附件" / "8.xlsx"


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], 1):
        val = min(prev, p[idx] * n / (n - rank + 1))
        q[idx] = val
        prev = val
    return q


def load_consumer():
    sh = pd.read_excel(P8, sheet_name="GBM SHAP Values")
    meta = ["data", "beer", "beer_id", "tasting_category_fine", "base_prediction"]
    comp = [c for c in sh.columns if c not in meta]
    d = sh[sh.data == "SHAP"].copy()
    out = d[["beer", "beer_id", "tasting_category_fine"]].copy()
    out["consumer_pred"] = d[comp].sum(axis=1).to_numpy() + d["base_prediction"].to_numpy()
    return out, d[comp], comp


def main():
    out = C.OUT_DIR / "consumer"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    scale = b.scale.copy()
    raw2clean = dict(zip(scale["原始名"].astype(str).str.strip(),
                         scale["列名"].astype(str).str.strip()))

    cons, shap, raw_cols = load_consumer()
    cons = cons.sort_values("beer").reset_index(drop=True)
    cons.to_csv(out / "consumer_predicted_appreciation.csv", index=False, encoding="utf-8-sig")

    # ---- 1) 目标层 ----
    sens = b.sens.reset_index()
    overall = "Overall appreciation (Trained panel)"
    oof = pd.read_csv(C.Q1_DIR / "results" / "oof_calibrated.csv", encoding="utf-8-sig")
    pr = pd.read_excel(P8, sheet_name="Beer Prices")
    price_cols = [c for c in pr.columns if c.startswith("Price_")]
    pr["price"] = pr[price_cols].mean(axis=1)

    m = cons.merge(sens[["beer", overall]], on="beer", how="left")
    m = m.merge(oof[["beer", overall]].rename(columns={overall: "panel_oof_pred"}),
                on="beer", how="left")
    m = m.merge(pr[["beer", "price"]], on="beer", how="left")
    m.to_csv(out / "consumer_target_level.csv", index=False, encoding="utf-8-sig")
    target_level = {
        "consumer_vs_panel_observed": float(spearmanr(m.consumer_pred, m[overall]).statistic),
        "consumer_vs_panel_oof_pred": float(spearmanr(m.consumer_pred, m.panel_oof_pred).statistic),
        "consumer_vs_price": float(spearmanr(m.dropna(subset=["price"]).consumer_pred,
                                             m.dropna(subset=["price"]).price).statistic),
        "panel_oof_vs_panel_observed": float(spearmanr(m.panel_oof_pred, m[overall]).statistic),
    }

    # ---- 2) 化合物层 ----
    X = b.X(C.VIEW_MAIN)
    y = m.set_index("beer").loc[X.index, "consumer_pred"].to_numpy()
    rows = []
    for c in X.columns:
        x = X[c].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        r, p = spearmanr(x[mask], y[mask]) if mask.sum() >= 10 else (np.nan, 1.0)
        rows.append({"compound": c, "rho_consumer": r, "p": p, "n": int(mask.sum())})
    uni = pd.DataFrame(rows)
    uni["q_consumer"] = bh_fdr(uni.p.to_numpy())
    uni["abs_rho_consumer"] = uni.rho_consumer.abs()
    uni = uni.sort_values("abs_rho_consumer", ascending=False).reset_index(drop=True)
    uni["consumer_rank"] = np.arange(1, len(uni) + 1)
    uni["chem_class"] = uni.compound.map(chem_class)
    uni.to_csv(out / "q2_consumer_univariate.csv", index=False, encoding="utf-8-sig")

    final = pd.read_csv(C.OUT_DIR / "final" / "q2_top3_final.csv", encoding="utf-8-sig")
    v3 = final[final.view == "V3"]
    ours = v3.groupby("compound").agg(
        n_rows=("target", "size"),
        n_stable=("status_final", lambda s: int((s == "stable").sum())),
        n_definitional=("status_final", lambda s: int((s == "definitional").sum())),
        max_abs_adj=("rho_adjusted", lambda s: float(pd.to_numeric(s).abs().max())),
        mean_boot=("boot_freq", "mean"),
    ).reset_index()
    comp = ours.merge(uni[["compound", "rho_consumer", "q_consumer", "abs_rho_consumer",
                           "consumer_rank", "chem_class"]], on="compound", how="left")

    shap_imp = {}
    for c in shap.columns:
        cl = raw2clean.get(c)
        if cl is None:
            continue
        shap_imp.setdefault(cl, []).append(shap[c].abs().to_numpy())
    comp["shap_mean_abs"] = comp.compound.map(
        {k: float(np.mean(np.mean(v, axis=0))) for k, v in shap_imp.items()})
    comp.to_csv(out / "q2_consumer_vs_ours.csv", index=False, encoding="utf-8-sig")

    d = comp.dropna(subset=["rho_consumer", "max_abs_adj"])
    rho_rank = float(spearmanr(d.max_abs_adj, d.abs_rho_consumer).statistic)
    d2 = comp.dropna(subset=["shap_mean_abs", "max_abs_adj"])
    rho_shap = float(spearmanr(d2.max_abs_adj, d2.shap_mean_abs).statistic)
    stable_set = set(v3[v3.status_final == "stable"].compound)
    top10, top20 = set(uni.head(10).compound), set(uni.head(20).compound)
    overlap = {
        "stable_unique": len(stable_set),
        "consumer_top10_overlap": len(stable_set & top10),
        "consumer_top20_overlap": len(stable_set & top20),
        "spearman_our_max_abs_vs_consumer_abs": rho_rank,
        "spearman_our_max_abs_vs_shap_mean": rho_shap,
    }

    # ---- 3) 类别层 ----
    grp_consumer = uni.groupby("chem_class").agg(
        n=("compound", "size"),
        max_abs_rho=("abs_rho_consumer", "max"),
        mean_abs_rho=("abs_rho_consumer", "mean"),
        n_sig=("q_consumer", lambda s: int((s < 0.05).sum())),
    ).reset_index()
    grp_shap = comp.groupby("chem_class").agg(
        shap_mean_abs=("shap_mean_abs", "mean")).reset_index()
    grp = grp_consumer.merge(grp_shap, on="chem_class", how="left")
    gpath = C.OUT_DIR / "group" / "q2_group_importance_class.csv"
    if gpath.exists():
        gimp = pd.read_csv(gpath, encoding="utf-8-sig")
        g_overall = gimp[gimp.target == overall].set_index("group")["delta_r2"].rename(
            "our_group_dr2_overall")
        g_mean = gimp.groupby("group")["delta_r2"].mean().rename("our_group_dr2_mean")
        grp["our_group_dr2_overall"] = grp.chem_class.map(g_overall)
        grp["our_group_dr2_mean"] = grp.chem_class.map(g_mean)
    else:
        print("WARNING: group results not found; skip class-level comparison.", flush=True)
        grp["our_group_dr2_overall"] = np.nan
        grp["our_group_dr2_mean"] = np.nan
    grp = grp.sort_values("max_abs_rho", ascending=False)
    grp.to_csv(out / "q2_consumer_group.csv", index=False, encoding="utf-8-sig")
    gg = grp.dropna(subset=["our_group_dr2_overall"])
    if len(gg) >= 3:
        src_rho = float(spearmanr(gg.max_abs_rho, gg.our_group_dr2_overall).statistic)
        src_rho_mean = float(spearmanr(gg.max_abs_rho, gg.our_group_dr2_mean).statistic)
    else:
        src_rho = src_rho_mean = float("nan")

    # ---- 4) overall Top-3 + 逐行方向 ----
    oa = v3[v3.target == overall].sort_values("rank")[
        ["rank", "compound", "status_final", "rho_adjusted", "boot_freq"]]
    oa = oa.merge(uni[["compound", "rho_consumer", "q_consumer", "consumer_rank"]],
                  on="compound", how="left")
    oa.to_csv(out / "q2_consumer_overall_top3.csv", index=False, encoding="utf-8-sig")

    allrows = v3.merge(uni[["compound", "rho_consumer", "q_consumer", "consumer_rank"]],
                       on="compound", how="left")
    allrows["consumer_direction"] = np.select(
        [allrows.q_consumer < 0.05, allrows.q_consumer >= 0.05],
        # 同向 = 候选在感官目标上的调整后方向与其消费者喜好相关方向一致
        [np.where(np.sign(allrows.rho_consumer) == np.sign(allrows.rho_adjusted),
                  "same_pos", "opposite"), "ns"],
        default="na")
    allrows.to_csv(out / "q2_top3_with_consumer.csv", index=False, encoding="utf-8-sig")
    st = allrows[allrows.status_final == "stable"]
    consumer_check = {
        "stable_rows": int(len(st)),
        "stable_consumer_same_pos": int((st.consumer_direction == "same_pos").sum()),
        "stable_consumer_opposite": int((st.consumer_direction == "opposite").sum()),
        "stable_consumer_ns": int((st.consumer_direction == "ns").sum()),
        "stable_consumer_missing": int(st.consumer_direction.isna().sum()),
    }
    opp = st[st.consumer_direction == "opposite"][["target", "compound", "rho_consumer"]]

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    classes = comp.chem_class.fillna("Unknown").unique()
    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(classes):
        sub = comp[comp.chem_class.fillna("Unknown") == cl]
        ax.scatter(sub.max_abs_adj, sub.abs_rho_consumer, s=26, alpha=0.8,
                   color=cmap(i % 10), label=cl)
    for r in comp[(comp.n_stable > 0) | (comp.consumer_rank <= 10)].itertuples():
        ax.annotate(r.compound, (r.max_abs_adj, r.abs_rho_consumer),
                    fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("候选在其入选感官维度内的最大 |调整后 ρ|")
    ax.set_ylabel("与消费者喜好预测分的 |Spearman ρ|")
    ax.set_title(f"Q2 候选化合物与消费者喜好锚点的对照（{len(comp)} 个候选；秩 ρ={rho_rank:.2f}）")
    ax.legend(fontsize=6, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig9_consumer_validation.png", dpi=300)
    plt.close(fig)

    summ = {
        "target_level": target_level,
        "compound_level": overlap,
        "class_level": {
            "spearman_consumer_max_vs_our_dr2_overall": src_rho,
            "spearman_consumer_max_vs_our_dr2_mean": src_rho_mean,
        },
        "consumer_top10": uni.head(10)[["compound", "rho_consumer", "q_consumer"]].to_dict("records"),
        "overall_top3": oa.to_dict("records"),
        "consumer_check_stable": consumer_check,
        "consumer_opposite_stable": opp.to_dict("records"),
    }
    (out / "step11_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summ["target_level"], ensure_ascii=False, indent=2))
    print(json.dumps(summ["compound_level"], ensure_ascii=False, indent=2))
    print(json.dumps(summ["class_level"], ensure_ascii=False, indent=2))
    print(json.dumps(summ["consumer_check_stable"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
