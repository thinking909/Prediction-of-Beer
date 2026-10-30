"""Q3 Step 6：生成评价（风格命中、real-vs-fake、MMD、覆盖率、多样性、相关结构）。

只对 n>=8 的风格做定量评价；稀有风格标记 low_confidence，不进入汇总。
输出：
  results/metrics_by_style.csv
  results/metrics_summary.csv
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

import q3_config as C
import dataio


def mmd2(X, Y):
    XY = np.vstack([X, Y])
    d2 = cdist(XY, XY, "sqeuclidean")
    med = np.median(d2[d2 > 0])
    gamma = 1.0 / max(med, 1e-12)
    kxx = np.exp(-gamma * cdist(X, X, "sqeuclidean")).mean()
    kyy = np.exp(-gamma * cdist(Y, Y, "sqeuclidean")).mean()
    kxy = np.exp(-gamma * cdist(X, Y, "sqeuclidean")).mean()
    return float(kxx + kyy - 2 * kxy)


def corr_dist(A, B):
    if A.shape[0] < 3 or B.shape[0] < 3:
        return float("nan")
    ca = pd.DataFrame(A).corr(method="spearman").to_numpy()
    cb = pd.DataFrame(B).corr(method="spearman").to_numpy()
    ca = np.nan_to_num(ca)
    cb = np.nan_to_num(cb)
    p = ca.shape[0]
    return float(np.linalg.norm(ca - cb) / np.sqrt(2 * p * (p - 1)))


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    X = b.chem[cols].copy()
    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    trm = folds.split == "outer_train"
    Xtr = X.loc[folds.loc[trm, "beer"].to_numpy()]
    style_tr = folds.loc[trm, "style_orig"].set_axis(Xtr.index)
    test = folds[~trm]
    Xte = X.loc[test.beer.to_numpy()]
    style_te = test.style_orig.set_axis(Xte.index)

    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    components = np.load(C.OUT_DIR / "pca_model.npz")["components"]
    Ztr = qt.transform(imp.transform(Xtr)) @ components.T
    Zte = qt.transform(imp.transform(Xte)) @ components.T
    # 真实性评估参考集：全部 250 款真实样本（生成器只在训练集上拟合）
    Xall = X.copy()
    style_all = b.style.astype(str).set_axis(Xall.index)
    Zall = qt.transform(imp.transform(Xall)) @ components.T
    gen = pd.read_csv(C.OUT_DIR / "constrained_profiles.csv", encoding="utf-8-sig")
    Zgen = qt.transform(imp.transform(gen[cols])) @ components.T
    ood = pd.read_csv(C.OUT_DIR / "ood_flags.csv", encoding="utf-8-sig")
    ood["ood_bool"] = ood["ood"].map({"True": True, "False": False, "na": np.nan})

    rows = []
    for s in sorted(gen["style"].unique()):
        idx_r = np.where(style_all.to_numpy() == s)[0]
        idx_g = gen.index[gen["style"] == s].to_numpy()
        n_r, n_g = len(idx_r), len(idx_g)
        if n_r < C.RARE_THRESHOLD:
            rows.append({"style": s, "n_real": n_r, "n_gen": n_g,
                         "low_confidence": True})
            continue
        Zr, Zg = Zall[idx_r], Zgen[idx_g]
        # 风格命中：one-vs-rest LR
        ytr = (style_tr.to_numpy() == s).astype(int)
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=C.SEED)
        clf.fit(Ztr, ytr)
        pg = clf.predict_proba(Zg)[:, 1]
        hit = float(np.mean(pg > 0.5))
        # real-vs-fake AUC：等样本量（fake 子采样到 n_real），重复 20 次取均值
        rng = np.random.default_rng(C.SEED)
        aucs = []
        clf2 = LogisticRegression(max_iter=5000, class_weight="balanced")
        for _ in range(20):
            idxf = rng.choice(len(Zg), len(Zr), replace=False)
            Xc = np.vstack([Zr, Zg[idxf]])
            yc = np.r_[np.zeros(len(Zr)), np.ones(len(Zr))]
            pv = cross_val_predict(
                clf2, Xc, yc,
                cv=StratifiedKFold(3, shuffle=True, random_state=int(rng.integers(1e9))),
                method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(yc, pv))
        auc = float(np.mean(aucs))
        # MMD
        m = mmd2(Zr, Zg)
        # coverage
        nn_real = NearestNeighbors(n_neighbors=2).fit(Zr)
        dr, _ = nn_real.kneighbors(Zr)
        thr = float(np.quantile(dr[:, 1], 0.95))
        nn1 = NearestNeighbors(n_neighbors=1).fit(Zg)
        d_rg, _ = nn1.kneighbors(Zr)
        coverage = float(np.mean(d_rg[:, 0] <= thr))
        # novelty / diversity
        nn2 = NearestNeighbors(n_neighbors=1).fit(Zr)
        d_gr, _ = nn2.kneighbors(Zg)
        real_pair = cdist(Zr, Zr)
        real_pair = real_pair[real_pair > 0]
        novelty = float(np.median(d_gr[:, 0]) / max(np.median(real_pair), 1e-12))
        gen_pair = cdist(Zg, Zg)
        gen_pair = gen_pair[gen_pair > 0]
        diversity = float(np.median(gen_pair) / max(np.median(real_pair), 1e-12))
        # 相关结构
        cdist_val = corr_dist(Xall.iloc[idx_r].to_numpy(), gen.iloc[idx_g][cols].to_numpy())
        o = ood[ood["style"] == s].ood_bool.mean()
        # outer-test 真实风格 AUC（可选）
        idx_te = np.where(style_te.to_numpy() == s)[0]
        auc_real = np.nan
        if len(idx_te) >= 1 and len(idx_te) < len(Zte):
            yte = (style_te.to_numpy() == s).astype(int)
            try:
                auc_real = float(roc_auc_score(yte, clf.predict_proba(Zte)[:, 1]))
            except Exception:
                pass
        rows.append({
            "style": s, "n_real": n_r, "n_gen": n_g,
            "low_confidence": bool(n_r < C.RARE_THRESHOLD),
            "style_hit_rate": hit, "style_mean_prob": float(np.mean(pg)),
            "style_auc_real": auc_real,
            "real_vs_fake_auc": auc, "mmd2": m,
            "coverage": coverage, "novelty_ratio": novelty,
            "diversity_ratio": diversity, "corr_dist": cdist_val,
            "ood_rate": float(o) if np.isfinite(o) else np.nan,
        })
    d = pd.DataFrame(rows)
    d.to_csv(C.OUT_DIR / "metrics_by_style.csv", index=False, encoding="utf-8-sig")

    main = d[~d.low_confidence].copy()
    summary = main[["style_hit_rate", "real_vs_fake_auc", "mmd2", "coverage",
                    "novelty_ratio", "diversity_ratio", "corr_dist", "ood_rate"]].agg(
        ["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "metric"})
    summary.to_csv(C.OUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    print(d.to_string(index=False))
    print("\nsummary (n>=8):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
