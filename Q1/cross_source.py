"""步骤7：跨源一致性验证。

用 7.csv 论坛评论描述符（经官方映射）作为独立数据源：
- 模型校准预测 vs 评论指标（外部效度）
- 观测感官分 vs 评论指标（两源一致性的上限）
逐维、逐方面计算 Spearman + BH-FDR。

输出：
  results/cross_source_pairs.csv
  results/cross_source_summary.csv
  results/cross_source_by_level.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import dataio

LEVEL_MAP = {
    "官方": "official",
    "官方(同描述符跨面)": "official_cross_aspect",
    "推断-高": "inferred_high",
    "推断-中": "inferred_mid",
    "推断-低": "inferred_low",
}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        val = min(prev, p[idx] * n / (n - rank + 1))
        q[idx] = val
        prev = val
    return q


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    Y = b.Y("model")
    targets = list(Y.columns)
    cal = pd.read_csv(C.OUT_DIR / "oof_calibrated.csv", index_col=0).loc[b.beer, targets]
    comments = b.comments

    mp = b.mapping
    mp = mp.rename(columns={mp.columns[0]: "comment", mp.columns[1]: "aspect",
                            mp.columns[2]: "descriptor", mp.columns[3]: "target",
                            mp.columns[4]: "level", mp.columns[5]: "note"})
    mp["level_en"] = mp["level"].map(LEVEL_MAP)
    mp = mp[mp.target.isin(targets) & mp.comment.isin(comments.columns)
            & mp.level_en.notna()].copy()

    rows = []
    for _, r in mp.iterrows():
        x = comments[r.comment].to_numpy()
        a = cal[r.target].to_numpy()
        o = Y[r.target].to_numpy()
        if np.std(x) == 0:
            continue
        rp, pp = spearmanr(a, x)
        ro, po = spearmanr(o, x)
        rows.append({
            "target": r.target, "comment_col": r.comment, "aspect": r.aspect,
            "descriptor": r.descriptor, "level": r.level_en,
            "spearman_pred": rp, "p_pred": pp,
            "spearman_obs": ro, "p_obs": po,
        })
    pairs = pd.DataFrame(rows)
    pairs["q_pred"] = bh_fdr(pairs.p_pred)
    pairs["q_obs"] = bh_fdr(pairs.p_obs)
    pairs.to_csv(C.OUT_DIR / "cross_source_pairs.csv", index=False)

    # 逐维聚合：同目标的多个方面 z-score 后取均值
    agg_rows = []
    for tgt, d in pairs.groupby("target"):
        z = []
        for _, r in d.iterrows():
            x = comments[r.comment_col]
            z.append((x - x.mean()) / x.std(ddof=0))
        ext = pd.concat(z, axis=1).mean(axis=1).to_numpy()
        rp, pp = spearmanr(cal[tgt].to_numpy(), ext)
        ro, po = spearmanr(Y[tgt].to_numpy(), ext)
        agg_rows.append({
            "target": tgt, "n_sources": len(d),
            "levels": "|".join(sorted(d.level.unique())),
            "best_level": d.level.iloc[0] if len(d) == 1 else "mixed",
            "spearman_pred": rp, "p_pred": pp,
            "spearman_obs": ro, "p_obs": po,
        })
    agg = pd.DataFrame(agg_rows)
    agg["q_pred"] = bh_fdr(agg.p_pred)
    agg["q_obs"] = bh_fdr(agg.p_obs)
    agg["sign_match"] = np.sign(agg.spearman_pred) == np.sign(agg.spearman_obs)
    agg.sort_values("spearman_obs", ascending=False).to_csv(
        C.OUT_DIR / "cross_source_summary.csv", index=False)

    # 按映射层级汇总（pair 级）
    by_level = []
    for lv, d in pairs.groupby("level"):
        by_level.append({
            "level": lv, "n_pairs": len(d),
            "pred_sig_pos": int(((d.q_pred < 0.05) & (d.spearman_pred > 0)).sum()),
            "pred_sig_neg": int(((d.q_pred < 0.05) & (d.spearman_pred < 0)).sum()),
            "obs_sig_pos": int(((d.q_obs < 0.05) & (d.spearman_obs > 0)).sum()),
            "obs_sig_neg": int(((d.q_obs < 0.05) & (d.spearman_obs < 0)).sum()),
            "median_pred": d.spearman_pred.median(),
            "median_obs": d.spearman_obs.median(),
        })
    by_level = pd.DataFrame(by_level)
    by_level.to_csv(C.OUT_DIR / "cross_source_by_level.csv", index=False)


if __name__ == "__main__":
    main()
