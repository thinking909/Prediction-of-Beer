"""Q2 Step 5：共识 Top-3 + 方法 Jaccard + 视图 Jaccard。

- V3：稳定性 + ET 置换 + PLS VIP；V2：稳定性 + PLS VIP（无 ET）
- Borda 平均排名，缺失=21；≥2 方法进 Top-20
- 缺陷维 Top-1 强制定义性；Top-2/3 只 exploratory
- 同社区（size≥3）尽量去重；状态为 stage1 初判
输出：Q2/results/consensus/*
"""
import json
from itertools import combinations

import numpy as np
import pandas as pd

import q2_config as C

DEFECT_MAP = {
    "Dimethyl sulfide": "Dimethyl sulfide",
    "Diacetyl": "Diacetyl",
    "4-vinyl guaiacol": "4-vinylguaiacol",
    "Trans-2-nonenal": "E-2-nonenal",
}
TOPN = 20


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def method_sets(d):
    """返回 {method: [top20 compounds]}，仅用候选池内。"""
    out = {}
    s = d[d.view.str.len() > 0] if "view" in d else d
    return out


def build_view(view, target, uni, stab, imp, nodes, has_perm):
    u = uni[(uni.view == view) & (uni.target == target)].set_index("compound")
    st = stab[(stab.view == view) & (stab.target == target)].copy()
    im = imp[(imp.view == view) & (imp.target == target)].copy()
    st = st.sort_values(["boot_freq", "mean_coef"], ascending=[False, False])
    pool = set(st.compound)
    im_pool = im[im.compound.isin(pool)].copy()

    sets = {"stability": list(st.head(TOPN).compound)}
    if has_perm:
        sets["permutation"] = list(im_pool.sort_values(
            "perm_importance", ascending=False).head(TOPN).compound)
    sets["vip"] = list(im_pool.sort_values(
        "pls_vip", ascending=False).head(TOPN).compound)

    rank_sum, n_methods = {}, {}
    for method, lst in sets.items():
        for rank, c in enumerate(lst, 1):
            rank_sum[c] = rank_sum.get(c, 0) + rank
            n_methods[c] = n_methods.get(c, 0) + 1
    cand = [(rank_sum[c] / n_methods[c], c, n_methods[c])
            for c in rank_sum if n_methods[c] >= 2]
    cand.sort(key=lambda z: (z[0], -z[2]))
    if not cand:  # 没达到 2 方法：取单方法最高频的，状态后置
        fallback = list(st.head(3).compound)
        cand = [(21.0, c, n_methods.get(c, 1)) for c in fallback]

    rows = []
    if target in DEFECT_MAP:
        defining = DEFECT_MAP[target]
        cand = [z for z in cand if z[1] != defining]
        top = [(0.0, defining, 3)] + cand[:2]
        for rank, (score, c, nm) in enumerate(top, 1):
            rows.append(make_row(view, target, rank, c, score, nm,
                                 st, im, u, nodes,
                                 "definitional" if rank == 1 else "exploratory"))
        return rows, sets

    # 同社区去重：top3 全在同一个 size≥3 社区时，尝试换掉第 3
    top = cand[:3]
    if len(top) == 3:
        comms = [nodes.get(c) for _, c, _ in top]
        cid = comms[0]
        size = sum(1 for x in nodes.values() if x == cid)
        if len(set(comms)) == 1 and cid is not None and size >= 3:
            outside = [z for z in cand[3:] if nodes.get(z[1]) != cid]
            if outside:
                top = top[:2] + [outside[0]]
    for rank, (score, c, nm) in enumerate(top, 1):
        rows.append(make_row(view, target, rank, c, score, nm,
                             st, im, u, nodes, None))
    return rows, sets


def make_row(view, target, rank, compound, score, nm, st, im, u, nodes, force):
    sr = st[st.compound == compound]
    ir = im[im.compound == compound]
    freq = float(sr.boot_freq.iloc[0]) if len(sr) else 0.0
    sign_cons = float(sr.sign_consistency.iloc[0]) if len(sr) else np.nan
    mean_coef = float(sr.mean_coef.iloc[0]) if len(sr) else np.nan
    pls_sign = float(ir.pls_sign.iloc[0]) if len(ir) else np.nan
    univ_rho = float(ir.univariate_rho.iloc[0]) if len(ir) else np.nan
    if np.isfinite(mean_coef) and mean_coef != 0:
        sign, sign_source = int(np.sign(mean_coef)), "lasso_coef"
    elif np.isfinite(pls_sign) and pls_sign != 0:
        sign, sign_source = int(pls_sign), "pls_loading"
    else:
        sign, sign_source = (int(np.sign(univ_rho)) if np.isfinite(univ_rho) else 0,
                             "univariate_rho")
    q = float(u.loc[compound, "q"]) if compound in u.index else np.nan
    rho = float(u.loc[compound, "rho"]) if compound in u.index else np.nan
    status = force
    if status == "definitional":
        sign, sign_source = 1, "definitional"
        sign_cons = 1.0
    if status is None:
        status = "stable_candidate" if (
            np.isfinite(q) and q < 0.05 and freq >= C.BOOT_FREQ and
            nm >= 2 and (not np.isfinite(sign_cons) or sign_cons >= C.SIGN_CONSISTENCY)
        ) else "exploratory"
    return {
        "view": view, "target": target, "rank": rank, "compound": compound,
        "status_stage1": status, "borda": score, "n_methods": nm,
        "boot_freq": freq, "sign_consistency": sign_cons,
        "sign": sign, "sign_source": sign_source,
        "univariate_q": q, "univariate_rho": rho,
        "community": nodes.get(compound),
    }


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "consensus"
    out.mkdir(parents=True, exist_ok=True)
    U = {
        "V3": pd.read_csv(C.OUT_DIR / "univariate" / "q2_univariate_v3.csv"),
        "V2": pd.read_csv(C.OUT_DIR / "univariate" / "q2_univariate_v2.csv"),
    }
    S = {
        "V3": pd.read_csv(C.OUT_DIR / "stability" / "q2_stability_selection_v3.csv"),
        "V2": pd.read_csv(C.OUT_DIR / "stability" / "q2_stability_selection_v2.csv"),
    }
    I = {
        "V3": pd.read_csv(C.OUT_DIR / "importance" / "q2_importance_v3.csv"),
        "V2": pd.read_csv(C.OUT_DIR / "importance" / "q2_importance_v2.csv"),
    }
    nodes_df = pd.read_csv(C.OUT_DIR / "graph" / "q2_graph_nodes.csv")
    nodes = dict(zip(nodes_df.compound, nodes_df.community))

    all_rows, jac_rows, view_sets = [], [], {}
    for view, has_perm in [("V3", True), ("V2", False)]:
        rows = []
        for t in C.SENSORY_TARGETS:
            r, sets = build_view(view, t, U[view], S[view], I[view], nodes, has_perm)
            rows.extend(r)
            view_sets[(view, t)] = [x["compound"] for x in r]
            for a, b in combinations(sets.keys(), 2):
                for k in (3, 10):
                    jac_rows.append({"view": view, "target": t,
                                     "method_a": a, "method_b": b, "k": k,
                                     "jaccard": jaccard(sets[a][:k], sets[b][:k])})
        d = pd.DataFrame(rows)
        d.to_csv(out / f"q2_top3_stage1_{view.lower()}.csv", index=False)
        all_rows.append(d)
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(out / "q2_top3_stage1.csv", index=False)
    jac = pd.DataFrame(jac_rows)
    jac.to_csv(out / "q2_method_jaccard.csv", index=False)

    # 视图 Jaccard（V3 vs V2，每目标 top3）
    vj = []
    for t in C.SENSORY_TARGETS:
        a = view_sets[("V3", t)]
        b = view_sets[("V2", t)]
        vj.append({"target": t, "jaccard_top3": jaccard(a, b),
                   "v3": " | ".join(a), "v2": " | ".join(b)})
    vj = pd.DataFrame(vj)
    vj.to_csv(out / "q2_view_jaccard.csv", index=False)

    summary = {
        "rows": {v: int((all_df.view == v).sum()) for v in ["V3", "V2"]},
        "status_stage1": {
            v: all_df[all_df.view == v].status_stage1.value_counts().to_dict()
            for v in ["V3", "V2"]
        },
        "targets_with_stable": {
            v: int(all_df[(all_df.view == v) &
                          (all_df.status_stage1 == "stable_candidate")]
                   .target.nunique())
            for v in ["V3", "V2"]
        },
        "targets_no_stable": {
            v: sorted(set(C.SENSORY_TARGETS) - set(
                all_df[(all_df.view == v) &
                       (all_df.status_stage1 == "stable_candidate")].target))
            for v in ["V3", "V2"]
        },
        "n_stable": {
            v: int((all_df[(all_df.view == v)].status_stage1 == "stable_candidate").sum())
            for v in ["V3", "V2"]
        },
        "method_jaccard": {
            v: jac[jac.view == v].groupby(["method_a", "method_b", "k"]).jaccard
            .agg(["mean", "median"]).round(3).reset_index().to_dict("records")
            for v in ["V3", "V2"]
        },
        "view_jaccard": {
            "mean": float(vj.jaccard_top3.mean()),
            "median": float(vj.jaccard_top3.median()),
            "n_zero": int((vj.jaccard_top3 == 0).sum()),
        },
    }
    (out / "step5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
