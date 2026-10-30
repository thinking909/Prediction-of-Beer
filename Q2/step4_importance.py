"""Q2 Step 4：ET MDI + 置换重要性 + PLS VIP。

- ET：全特征 MDI 取 Top-20，再 5 折 ×10 次置换（在 Top-20 子模型上）
- PLS：候选池内选 n_components，算 VIP 与载荷符号
- 符号：ET 无符号，用单变量 Spearman 补充；PLS 用载荷符号
输出：Q2/results/importance/*
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import q2_config as C
import dataio

K_MDI = 20
N_FOLDS = 3
N_REPEATS = 5
EXAMPLES = ["Acidity", "Hops aroma - overall", "Esters aroma - floral", "Dimethyl sulfide"]


def pls_vip(Z, y, seed=C.SEED):
    best = None
    for k in [2, 3, 5, 8, 12]:
        kf = KFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y))
        for tr, te in kf.split(Z):
            m = PLSRegression(n_components=k, scale=False).fit(Z[tr], y[tr])
            pred[te] = m.predict(Z[te]).ravel()
        r2 = r2_score(y, pred)
        if best is None or r2 > best[1]:
            best = (k, r2)
    k = best[0]
    m = PLSRegression(n_components=k, scale=False).fit(Z, y)
    t = m.x_scores_
    w = m.x_weights_
    q = m.y_loadings_
    s = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)
    vip = np.sqrt(w.shape[0] * ((w ** 2) / (w ** 2).sum(axis=0) * s).sum(axis=1) / s.sum())
    sign = np.sign((w * s).sum(axis=1))
    return vip, sign, k, best[1]


def run_view(b, view_key, view_label, cand_file, seed, do_perm=True):
    X = b.X(view_key)
    Y = b.sens[C.SENSORY_TARGETS]
    cand = pd.read_csv(cand_file)
    rows = []
    t0 = time.time()
    for t in Y.columns:
        y = Y[t].to_numpy()
        Xv = X.to_numpy()
        imp_full = SimpleImputer(strategy="median").fit(Xv)
        Z = imp_full.transform(Xv)
        et_full = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 1})
        et_full.fit(Z, y)
        mdi = et_full.feature_importances_
        idx = np.argsort(-mdi)[:K_MDI]

        perm = np.zeros(X.shape[1])
        if do_perm:
            for tr, te in KFold(N_FOLDS, shuffle=True, random_state=seed).split(X):
                im = SimpleImputer(strategy="median").fit(X.iloc[tr])
                A, B = im.transform(X.iloc[tr]), im.transform(X.iloc[te])
                et = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 1})
                et.fit(A[:, idx], y[tr])
                pi = permutation_importance(et, B[:, idx], y[te], scoring="r2",
                                            n_repeats=N_REPEATS, random_state=seed,
                                            n_jobs=1)
                perm[idx] += pi.importances_mean
            perm /= N_FOLDS

        # 单变量符号（ET 无符号）
        rho = np.full(X.shape[1], np.nan)
        for j, c in enumerate(X.columns):
            x = Xv[:, j]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 10:
                rho[j] = spearmanr(x[mask], y[mask]).statistic

        # PLS VIP（候选池内）
        feats = [c for c in cand.loc[cand.target == t, "compound"] if c in X.columns]
        vip = np.full(X.shape[1], np.nan)
        psign = np.full(X.shape[1], np.nan)
        if len(feats) >= 2:
            Xc = X[feats].to_numpy()
            imc = SimpleImputer(strategy="median").fit(Xc)
            scc = StandardScaler().fit(imc.transform(Xc))
            Zc = scc.transform(imc.transform(Xc))
            v, sg, k, _ = pls_vip(Zc, y, seed)
            pos = [X.columns.get_loc(c) for c in feats]
            vip[pos] = v
            psign[pos] = sg

        for j, c in enumerate(X.columns):
            rows.append({
                "view": view_label, "target": t, "compound": c,
                "mdi": float(mdi[j]),
                "mdi_top20": bool(j in idx),
                "perm_importance": float(perm[j]),
                "univariate_rho": float(rho[j]),
                "pls_vip": float(vip[j]),
                "pls_sign": float(psign[j]) if np.isfinite(psign[j]) else np.nan,
                "in_candidate_pool": c in feats,
            })
        print(f"  [{view_label}] {t}: MDI top20, perm={'off' if not do_perm else f'{N_FOLDS}x{N_REPEATS}'}", flush=True)
    out = pd.DataFrame(rows)
    print(f"[{view_label}] done in {time.time()-t0:.1f}s", flush=True)
    return out


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "importance"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    print("[1/2] V3", flush=True)
    i3 = run_view(b, C.VIEW_MAIN, "V3",
                  C.OUT_DIR / "univariate" / "q2_candidates_v3.csv", C.SEED)
    i3.to_csv(out / "q2_importance_v3.csv", index=False)
    print("[2/2] V2", flush=True)
    i2 = run_view(b, C.VIEW_CHECK, "V2",
                  C.OUT_DIR / "univariate" / "q2_candidates_v2.csv", C.SEED + 1,
                  do_perm=False)
    i2.to_csv(out / "q2_importance_v2.csv", index=False)

    summary = {}
    for name, d in [("v3", i3), ("v2", i2)]:
        per = d.groupby("target").agg(
            top_perm=("perm_importance", "max"),
            n_perm_gt0=("perm_importance", lambda x: int((x > 0).sum())),
            n_vip_gt1=("pls_vip", lambda x: int((x > 1).sum())),
        )
        summary[name] = {
            "targets": int(per.shape[0]),
            "median_top_perm": float(per.top_perm.median()),
            "median_n_perm_gt0": float(per.n_perm_gt0.median()),
            "median_n_vip_gt1": float(per.n_vip_gt1.median()),
        }
    examples = {}
    for t in EXAMPLES:
        d = i3[i3.target == t].sort_values("perm_importance", ascending=False).head(5)
        examples[t] = d[["compound", "mdi", "perm_importance", "pls_vip",
                         "univariate_rho"]].round(4).to_dict("records")
    summary["examples_v3"] = examples
    (out / "step4_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
