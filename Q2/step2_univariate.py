"""Q2 Step 2：全量单变量先验 + 6.csv 核对 + 候选池。

- V3（70）与 V2（227）分别做 50 目标的 Spearman，每目标内 BH-FDR
- 6.csv（44×50）与原始 pairwise Spearman 核对
- 候选池：每目标 |rho| 前 20 + 同社区邻居（社区 size≥3），最多 60
输出：Q2/results/univariate/*
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import q2_config as C
import dataio

TOP_N = 20
POOL_CAP = 60
COMM_MIN = 3
EXAMPLES = ["Acidity", "Hops aroma - overall", "Esters aroma - floral", "Dimethyl sulfide"]


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


def univariate(X, Y, view):
    rows = []
    for t in Y.columns:
        y = Y[t].to_numpy()
        ps, tmp = [], []
        for c in X.columns:
            x = X[c].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                r, p = np.nan, 1.0
            else:
                r, p = spearmanr(x[mask], y[mask])
            tmp.append({"view": view, "target": t, "compound": c,
                        "rho": r, "p": p, "n": int(mask.sum())})
            ps.append(p)
        q = bh_fdr(ps)
        for i, row in enumerate(tmp):
            row["q"] = q[i]
            row["abs_rho"] = abs(row["rho"])
            row["fdr_sig"] = q[i] < C.FDR_Q
            rows.append(row)
    return pd.DataFrame(rows)


def check_6csv(m6, chem, sens, out):
    rows = []
    for a in m6.index:
        for b in m6.columns:
            x, y = chem[a], sens[b]
            mask = x.notna() & y.notna()
            r = spearmanr(x[mask], y[mask]).statistic if mask.sum() >= 10 else np.nan
            rows.append({"chemical": a, "sensory": b,
                         "official_6csv": float(m6.loc[a, b]),
                         "recomputed": float(r),
                         "abs_diff": abs(float(m6.loc[a, b]) - float(r)),
                         "sign_match": np.sign(m6.loc[a, b]) == np.sign(r)})
    d = pd.DataFrame(rows)
    d.to_csv(out / "q2_6csv_check.csv", index=False)
    return {
        "n_pairs": int(len(d)),
        "median_abs_diff": float(d.abs_diff.median()),
        "max_abs_diff": float(d.abs_diff.max()),
        "sign_agreement": float(d.sign_match.mean()),
        "n_gt_0.01": int((d.abs_diff > 0.01).sum()),
    }


def bh_sig_targets(X, Y):
    """逐目标对 X 的列做 BH-FDR，返回至少 1 个显著指标的目标准数。"""
    n = 0
    for t in Y.columns:
        y = Y[t].to_numpy(float)
        ps = []
        for c in X.columns:
            x = X[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            ps.append(spearmanr(x[m], y[m]).pvalue if m.sum() >= 10 else 1.0)
        n += bool((bh_fdr(ps) < C.FDR_Q).any())
    return int(n)


def caliber_check(b, X3, Y, u3, u2, out):
    """核对“70 列绝对量主视图”在不同数值来源/校正集合下的显著目标数。

    同为 70 个字段，数值版本不同（附件1原始 log 列 vs 附件8校准集）
    或校正集合不同（BH70 vs 先 BH227 再回取）都会改变计数，必须显式记录。
    """
    rows = [{
        "variant": "A_1csv70_BH70",
        "n_features": int(X3.shape[1]),
        "n_targets_sig": int((u3.groupby("target").fdr_sig.sum() > 0).sum()),
        "note": "附件1的70列原始值（log10，保留缺失，pairwise complete），逐目标BH70；本管线主口径",
    }]
    sub = u2[u2.compound.isin(X3.columns)]
    rows.append({
        "variant": "B_227_BH_then_70",
        "n_features": 227,
        "n_targets_sig": int((sub[sub.q < C.FDR_Q].groupby("target").size() > 0).sum()),
        "note": "先在227列上做BH227，再回取这70列的显著结果",
    })
    try:
        cal = pd.ExcelFile(C.ROOT / "附件" / "8.xlsx").parse("Calibrated chemical dataset")
        ren = dict(zip(b.scale["原始名"], b.scale["列名"]))
        cal = cal.rename(columns=ren).set_index(b.chem.index.name).reindex(X3.index)
        cal = cal.replace([np.inf, -np.inf], np.nan)
        cols70 = [c for c in X3.columns if c in cal.columns]
        n_c = bh_sig_targets(cal[cols70], Y) if len(cols70) == X3.shape[1] else np.nan
        rows.append({
            "variant": "C_8xlsx_cal_70_BH70",
            "n_features": len(cols70),
            "n_targets_sig": n_c,
            "note": "附件8“Calibrated chemical dataset”的70列，BH70；仅作口径核对，不作主口径",
        })
    except Exception as exc:  # pragma: no cover
        rows.append({"variant": "C_8xlsx_cal_70_BH70", "n_features": np.nan,
                     "n_targets_sig": np.nan, "note": f"跳过：{exc}"})
    d = pd.DataFrame(rows)
    d.to_csv(out / "q2_view70_caliber.csv", index=False, encoding="utf-8-sig")
    return d.to_dict("records")


def build_candidates(uni, X, nodes):
    comm_map = nodes.set_index("compound")["community"].to_dict()
    comm_size = nodes.groupby("community").size().to_dict()
    pool = []
    for t, d in uni.groupby("target"):
        d = d.sort_values("abs_rho", ascending=False)
        top = d.head(TOP_N).copy()
        top["source"] = "top20"
        selected = list(top.compound)
        neigh = []
        for _, r in top.iterrows():
            c = r.compound
            if c not in comm_map:
                continue
            cid = comm_map[c]
            if comm_size.get(cid, 0) < COMM_MIN:
                continue
            same = nodes[(nodes.community == cid) & (~nodes.is_derived)]
            for n in same.compound:
                if n in X.columns and n not in selected and n not in neigh:
                    neigh.append(n)
        if neigh:
            nd = d[d.compound.isin(neigh)].copy()
            nd["source"] = "community_neighbor"
            top = pd.concat([top, nd], ignore_index=True)
        # 保 top20，其余按 |rho| 补到 cap
        top = top.sort_values(["source", "abs_rho"], ascending=[True, False])
        fixed = top[top.source == "top20"]
        rest = top[top.source != "top20"].sort_values("abs_rho", ascending=False)
        keep = pd.concat([fixed, rest.head(max(0, POOL_CAP - len(fixed)))])
        keep["community"] = keep.compound.map(comm_map)
        keep["pool_size"] = len(keep)
        pool.append(keep)
    return pd.concat(pool, ignore_index=True)


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "univariate"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    X3 = b.X("V3_absolute")
    X2 = b.X("V2_scale_aware")
    Y = b.sens[C.SENSORY_TARGETS]
    m6 = pd.read_csv(C.DATA_DIR / "6_化学感官参数Spearman相关矩阵.csv",
                     index_col=0, encoding="utf-8-sig")
    nodes = pd.read_csv(C.OUT_DIR / "graph" / "q2_graph_nodes.csv")

    print("[1/4] univariate V3", flush=True)
    u3 = univariate(X3, Y, "V3")
    u3.to_csv(out / "q2_univariate_v3.csv", index=False)
    print("[2/4] univariate V2", flush=True)
    u2 = univariate(X2, Y, "V2")
    u2.to_csv(out / "q2_univariate_v2.csv", index=False)
    print("[3/4] 6.csv check", flush=True)
    c6 = check_6csv(m6, b.chem, b.sens, out)
    print("[4/4] candidates", flush=True)
    cand3 = build_candidates(u3, X3, nodes)
    cand3.to_csv(out / "q2_candidates_v3.csv", index=False)
    cand2 = build_candidates(u2, X2, nodes)
    cand2.to_csv(out / "q2_candidates_v2.csv", index=False)
    print("[5/5] 70 列口径核对", flush=True)
    caliber = caliber_check(b, X3, Y, u3, u2, out)
    for r in caliber:
        print(f"  {r['variant']}: {r['n_targets_sig']}/50", flush=True)

    def fdr_summary(u):
        d = u.groupby("target").agg(n_sig=("fdr_sig", "sum"),
                                    min_p=("p", "min"),
                                    top_rho=("abs_rho", "max"))
        return {
            "targets_with_any_sig": int((d.n_sig > 0).sum()),
            "n_sig_median": float(d.n_sig.median()),
            "n_sig_min": int(d.n_sig.min()),
            "n_sig_max": int(d.n_sig.max()),
            "top_rho_median": float(d.top_rho.median()),
        }

    def pool_summary(c):
        d = c.groupby("target").size()
        return {"min": int(d.min()), "median": float(d.median()),
                "max": int(d.max())}

    examples = {}
    for t in EXAMPLES:
        d = u3[u3.target == t].sort_values("abs_rho", ascending=False).head(5)
        examples[t] = d[["compound", "rho", "q"]].round(4).to_dict("records")

    summary = {
        "v3": {"n_features": X3.shape[1], "fdr": fdr_summary(u3),
               "pool": pool_summary(cand3), "examples": examples},
        "v2": {"n_features": X2.shape[1], "fdr": fdr_summary(u2),
               "pool": pool_summary(cand2)},
        "six_csv_check": c6,
        "view70_caliber": caliber,
        "params": {"top_n": TOP_N, "pool_cap": POOL_CAP, "comm_min": COMM_MIN},
    }
    (out / "step2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
