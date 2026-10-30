"""Q2 Step 3：稳定性选择（Lasso/ElasticNet + bootstrap）。

- V3 主视图、V2 稳健性视图
- 每个目标：候选池内 ElasticNetCV 选 alpha/l1，再 bootstrap B=200
- 记录入选频率、平均系数、符号一致性
输出：Q2/results/stability/*
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler

import q2_config as C
import dataio

B_BOOT = 200
EXAMPLES = ["Acidity", "Hops aroma - overall", "Esters aroma - floral", "Dimethyl sulfide"]


def run_view(b, view_key, view, cand_file, seed):
    X_all = b.X(view_key)
    Y = b.sens[C.SENSORY_TARGETS]
    cand = pd.read_csv(cand_file)
    rows = []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for t in Y.columns:
        feats = list(cand.loc[cand.target == t, "compound"])
        feats = [c for c in feats if c in X_all.columns]
        if not feats:
            continue
        X = X_all[feats].to_numpy()
        y = Y[t].to_numpy()
        n = len(y)
        imp = SimpleImputer(strategy="median").fit(X)
        sc = StandardScaler().fit(imp.transform(X))
        Z = sc.transform(imp.transform(X))
        cv = ElasticNetCV(l1_ratio=[0.5, 0.7, 0.9, 1.0], cv=5,
                          n_alphas=50, max_iter=5000, random_state=0,
                          n_jobs=1).fit(Z, y)
        alpha, l1 = cv.alpha_, cv.l1_ratio_
        cnt = np.zeros(len(feats))
        pos = np.zeros(len(feats))
        neg = np.zeros(len(feats))
        coefs = []
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, n)
            imp_b = SimpleImputer(strategy="median").fit(X[idx])
            sc_b = StandardScaler().fit(imp_b.transform(X[idx]))
            Zb = sc_b.transform(imp_b.transform(X[idx]))
            en = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000,
                            random_state=0).fit(Zb, y[idx])
            c = en.coef_
            sel = c != 0
            cnt += sel
            pos += (c > 0) & sel
            neg += (c < 0) & sel
            coefs.append(c)
        mean_coef = np.mean(coefs, axis=0)
        for j, c in enumerate(feats):
            rows.append({
                "view": view, "target": t, "compound": c,
                "boot_freq": cnt[j] / B_BOOT,
                "mean_coef": mean_coef[j],
                "sign_consistency": (max(pos[j], neg[j]) / cnt[j]) if cnt[j] else np.nan,
                "n_selected": int(cnt[j]),
                "alpha": alpha, "l1_ratio": l1,
                "pool_size": len(feats),
            })
        print(f"  [{view}] {t}: {len(feats)} feats, alpha={alpha:.4g}, "
              f"{len([x for x in coefs if np.any(x != 0)])} nonempty bootstraps",
              flush=True)
    out = pd.DataFrame(rows)
    print(f"[{view}] done in {time.time()-t0:.1f}s", flush=True)
    return out


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "stability"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    print("[1/2] V3", flush=True)
    s3 = run_view(b, C.VIEW_MAIN, "V3",
                  C.OUT_DIR / "univariate" / "q2_candidates_v3.csv", C.SEED)
    s3.to_csv(out / "q2_stability_selection_v3.csv", index=False)
    print("[2/2] V2", flush=True)
    s2 = run_view(b, C.VIEW_CHECK, "V2",
                  C.OUT_DIR / "univariate" / "q2_candidates_v2.csv", C.SEED + 1)
    s2.to_csv(out / "q2_stability_selection_v2.csv", index=False)

    summary = {}
    for name, d in [("v3", s3), ("v2", s2)]:
        per = d.groupby("target").agg(
            n_selected=("boot_freq", lambda x: int((x >= C.BOOT_FREQ).sum())),
            max_freq=("boot_freq", "max"),
            n_any=("n_selected", lambda x: int((x > 0).sum())),
            pool_size=("pool_size", "first"),
        )
        summary[name] = {
            "targets_with_stable": int((per.n_selected > 0).sum()),
            "median_stable_per_target": float(per.n_selected.median()),
            "targets_with_any_selection": int((per.n_any > 0).sum()),
            "median_pool_size": float(per.pool_size.median()),
            "max_freq_median": float(per.max_freq.median()),
        }
    # 例子
    examples = {}
    for t in EXAMPLES:
        d = s3[s3.target == t].sort_values("boot_freq", ascending=False).head(5)
        examples[t] = d[["compound", "boot_freq", "mean_coef", "sign_consistency"]].round(4).to_dict("records")
    summary["examples_v3"] = examples
    (out / "step3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
