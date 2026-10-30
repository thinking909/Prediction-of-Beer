"""给共线组 drop 加随机组对照，估计“组贡献”是否显著高于随机删同样多的列。"""
import sys
sys.path.insert(0, '../Q1')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import config as q1c
import dataio
import q2_config as C

RNG = np.random.default_rng(C.SEED)
N_RAND = 20


def evaluate(X, y, drop_cols, folds):
    cols = [c for c in X.columns if c not in drop_cols]
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        Xtr, Xte = X.iloc[tr][cols], X.iloc[te][cols]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        et = ExtraTreesRegressor(**{**q1c.FINAL_ET_PARAMS, "n_jobs": 4})
        et.fit(imp.transform(Xtr), y[tr])
        oof[te] = et.predict(imp.transform(Xte))
    return r2_score(y, oof), float(np.sqrt(mean_squared_error(y, oof)))


def main():
    b = dataio.load_all()
    X = b.X(C.VIEW_MAIN)
    final = pd.read_csv(C.OUT_DIR / "final" / "q2_top3_final.csv")
    v3 = final[final.view == "V3"]
    diag = pd.read_csv(C.OUT_DIR / "final" / "q2_rescue_diagnostic.csv")
    ts = set(diag.target) - {"4-vinyl guaiacol", "Diacetyl", "Dimethyl sulfide", "Trans-2-nonenal"}
    out = []
    for t in diag.target:
        sub = v3[v3.target == t].sort_values("rank")
        top = list(sub.compound)
        if t not in ts:
            out.append({"target": t, "group_drop_dr2": np.nan, "rand_dr2_mean": np.nan,
                        "rand_dr2_p95": np.nan, "group_p_vs_random": np.nan,
                        "group_sig_vs_random": False})
            continue
        y = b.sens[t].to_numpy()
        ti = C.SENSORY_TARGETS.index(t)
        folds = list(KFold(3, shuffle=True, random_state=C.SEED + ti).split(X))
        base_r2, _ = evaluate(X, y, [], folds)
        group = set(top)
        for c in top:
            x = X[c].to_numpy()
            for c2 in X.columns:
                if c2 in group:
                    continue
                x2 = X[c2].to_numpy()
                m = np.isfinite(x) & np.isfinite(x2)
                if m.sum() >= 10 and abs(spearmanr(x[m], x2[m]).statistic) >= 0.7:
                    group.add(c2)
        grp_r2, _ = evaluate(X, y, list(group), folds)
        obs = base_r2 - grp_r2
        pool = [c for c in X.columns if c not in group]
        rnd = []
        for _ in range(N_RAND):
            pick = list(RNG.choice(pool, size=len(group), replace=False))
            r, _ = evaluate(X, y, pick, folds)
            rnd.append(base_r2 - r)
        rnd = np.array(rnd)
        p = (1 + int((rnd >= obs).sum())) / (1 + N_RAND)
        out.append({"target": t, "group_drop_dr2": obs, "rand_dr2_mean": float(rnd.mean()),
                    "rand_dr2_p95": float(np.quantile(rnd, 0.95)),
                    "group_p_vs_random": p, "group_sig_vs_random": bool(p < 0.05)})
        print(f"{t:<38} obs={obs:+.4f} rand={rnd.mean():+.4f} p95={np.quantile(rnd,0.95):+.4f} p={p:.3f}",
              flush=True)
    d = pd.DataFrame(out)
    d.to_csv(C.OUT_DIR / "final" / "q2_rescue_random_control.csv", index=False, encoding="utf-8-sig")
    print("\nrescue (group sig vs random):", int(d.group_sig_vs_random.sum()), "/", len(ts))


if __name__ == "__main__":
    main()

