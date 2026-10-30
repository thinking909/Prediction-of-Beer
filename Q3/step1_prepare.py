"""Q3 Step 1：数据准备、共享 PCA、风格指纹、OOD 参考。

输出：
  results/style_stats.csv
  results/folds.csv
  results/pca_cv.csv
  results/pca_reconstruction.csv
  results/pca_model.npz
  results/pca_meta.json
  results/style_baselines.csv
  results/compound_meta.csv
  results/style_fingerprint.csv
  results/ood_reference.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer

import q3_config as C
import dataio

PROXIES = {
    "Color", "pH", "Protein", "Caloric value", "Carbon dioxide", "Sugars",
    "Unfermentables", "Iron", "Betaglucan", "Ammonia", "Sulfur dioxide",
}


def fit_preprocess(Xtr):
    imp = SimpleImputer(strategy="median").fit(Xtr)
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(Xtr)),
        output_distribution="normal",
        subsample=len(Xtr),
        random_state=C.SEED,
    ).fit(imp.transform(Xtr))
    return imp, qt


def transform(imp, qt, X):
    return qt.transform(imp.transform(X))


def pca_cv_table(Xtr, style_tr, k_range):
    rows = []
    skf = StratifiedKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED)
    for k in k_range:
        rmses = []
        for itr, iva in skf.split(np.zeros(len(Xtr)), style_tr.to_numpy()):
            imp, qt = fit_preprocess(Xtr.iloc[itr])
            pca = PCA(n_components=k).fit(transform(imp, qt, Xtr.iloc[itr]))
            Zva_full = transform(imp, qt, Xtr.iloc[iva])
            Zhat = pca.inverse_transform(pca.transform(Zva_full))
            rmses.append(float(np.sqrt(np.mean((Zhat - Zva_full) ** 2))))
        rows.append({"k": k, "rmse_std": float(np.mean(rmses)),
                     "rmse_sd": float(np.std(rmses))})
    d = pd.DataFrame(rows)
    best = d.rmse_std.min()
    cand = d[d.rmse_std <= best * 1.02]
    return d, int(cand.k.min())


def reconstruction_cv(Xtr, style_tr, k):
    skf = StratifiedKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED)
    sse = np.zeros(Xtr.shape[1])
    sst = np.zeros(Xtr.shape[1])
    for itr, iva in skf.split(np.zeros(len(Xtr)), style_tr.to_numpy()):
        imp, qt = fit_preprocess(Xtr.iloc[itr])
        pca = PCA(n_components=k).fit(transform(imp, qt, Xtr.iloc[itr]))
        Zva_full = transform(imp, qt, Xtr.iloc[iva])
        Zhat = pca.inverse_transform(pca.transform(Zva_full))
        sse += np.sum((Zhat - Zva_full) ** 2, axis=0)
        sst += np.sum((Zva_full - Zva_full.mean(axis=0)) ** 2, axis=0)
    r2 = 1 - sse / np.maximum(sst, 1e-12)
    rmse = np.sqrt(sse / max(len(Xtr), 1))
    return r2, rmse


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    X = b.chem[cols].copy()
    style = b.style.copy()
    counts = style.value_counts()
    style_fold = style.where(style.map(counts) >= C.RARE_THRESHOLD, "Other")

    # ---- folds：先合并稀有风格，再分层 ----
    idx = np.arange(len(X))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=C.OUTER_TEST_SIZE,
                                 random_state=C.SEED)
    tr, te = next(sss.split(idx, style_fold.to_numpy()))
    split = np.array(["outer_train"] * len(idx), dtype=object)
    split[te] = "outer_test"
    inner = np.full(len(idx), -1)
    skf = StratifiedKFold(n_splits=C.N_INNER, shuffle=True, random_state=C.SEED)
    for k, (itr, iva) in enumerate(skf.split(tr, style_fold.iloc[tr].to_numpy())):
        inner[tr[iva]] = k
    folds = pd.DataFrame({"beer": b.beer, "style_orig": style.to_numpy(),
                          "style_fold": style_fold.to_numpy(),
                          "split": split, "inner_fold": inner})
    folds.to_csv(C.OUT_DIR / "folds.csv", index=False, encoding="utf-8-sig")

    Xtr = X.iloc[tr]
    style_tr = style_fold.iloc[tr]

    # ---- 共享 PCA：内层 CV 选 k ----
    pca_cv, k_choice = pca_cv_table(Xtr, style_tr, C.K_PCA_RANGE)
    pca_cv.to_csv(C.OUT_DIR / "pca_cv.csv", index=False, encoding="utf-8-sig")
    r2, rmse = reconstruction_cv(Xtr, style_tr, k_choice)

    imp, qt = fit_preprocess(Xtr)
    Ztr = transform(imp, qt, Xtr)
    pca = PCA(n_components=k_choice).fit(Ztr)
    Ztr_pca = pca.transform(Ztr)

    recon = pd.DataFrame({
        "compound": cols,
        "r2_cv": r2,
        "rmse_std_cv": rmse,
        "is_key": [c in C.KEY_COMPOUNDS for c in cols],
        "low_recon": r2 < 0.90,
    }).sort_values("r2_cv")
    recon.to_csv(C.OUT_DIR / "pca_reconstruction.csv", index=False, encoding="utf-8-sig")

    import joblib
    joblib.dump({"imputer": imp, "quantile": qt}, C.OUT_DIR / "preprocess.pkl")
    np.savez(
        C.OUT_DIR / "pca_model.npz",
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        k=k_choice,
    )
    (C.OUT_DIR / "pca_meta.json").write_text(json.dumps({
        "k": int(k_choice),
        "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cum_explained": float(np.sum(pca.explained_variance_ratio_)),
        "n_train": int(len(Xtr)),
        "columns": cols,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- style stats / baselines / meta ----
    stats = []
    for s, idxs in style.groupby(style).groups.items():
        pos = [X.index.get_loc(i) for i in idxs]
        stats.append({"style": s, "n": len(pos),
                      "n_missing": int(X.iloc[pos].isna().sum().sum()),
                      "missing_rate": float(X.iloc[pos].isna().mean().mean()),
                      "fold_group": "Other" if len(pos) < C.RARE_THRESHOLD else s})
    pd.DataFrame(stats).sort_values("n", ascending=False).to_csv(
        C.OUT_DIR / "style_stats.csv", index=False, encoding="utf-8-sig")

    base_rows = []
    for s, idxs in style.groupby(style).groups.items():
        pos = [X.index.get_loc(i) for i in idxs]
        sub = X.iloc[pos]
        for c in cols:
            v = sub[c].dropna()
            if len(v) == 0:
                continue
            base_rows.append({"style": s, "compound": c, "n": len(v),
                              "mean": float(v.mean()), "q05": float(v.quantile(0.05)),
                              "q50": float(v.median()), "q95": float(v.quantile(0.95))})
    pd.DataFrame(base_rows).to_csv(C.OUT_DIR / "style_baselines.csv",
                                   index=False, encoding="utf-8-sig")

    meta = b.scale.set_index("列名").reindex(cols).reset_index()
    meta = meta.rename(columns={"列名": "compound"})
    meta["is_key"] = meta["compound"].isin(C.KEY_COMPOUNDS)
    meta["is_defect"] = meta["compound"].isin(
        ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"])
    meta["is_log"] = meta["尺度类别"].astype(str).str.contains("log10")
    meta.to_csv(C.OUT_DIR / "compound_meta.csv", index=False, encoding="utf-8-sig")

    # ---- style fingerprint：标准化化合物空间 ----
    Zstd = pd.DataFrame(Ztr, index=Xtr.index, columns=cols)
    global_mean = Zstd.mean(axis=0)
    fp_rows = []
    for s, idxs in style_tr.groupby(style_tr).groups.items():
        if len(idxs) < C.RARE_THRESHOLD:
            continue
        sub = Zstd.loc[idxs]
        score = (sub.mean(axis=0) - global_mean).abs()
        cand = [c for c in score.sort_values(ascending=False).index if c not in PROXIES]
        for rank, c in enumerate(cand[:15], 1):
            fp_rows.append({"style": s, "rank": rank, "compound": c,
                            "score": float(score[c])})
    pd.DataFrame(fp_rows).to_csv(C.OUT_DIR / "style_fingerprint.csv",
                                 index=False, encoding="utf-8-sig")

    # ---- OOD reference in PCA space ----
    global_cov = np.cov(Ztr_pca, rowvar=False)
    ood = {}
    for s, idxs in style_tr.groupby(style_tr).groups.items():
        if len(idxs) < C.RARE_THRESHOLD:
            continue
        pos = [Xtr.index.get_loc(i) for i in idxs]
        Zs = Ztr_pca[pos]
        mu = Zs.mean(axis=0)
        n = len(pos)
        lam = C.N0_SHRINK / (C.N0_SHRINK + n)
        cov = (1 - lam) * np.cov(Zs, rowvar=False) + lam * global_cov
        cov = cov + np.eye(cov.shape[0]) * 1e-6
        prec = np.linalg.pinv(cov)
        d = np.array([float((z - mu) @ prec @ (z - mu)) for z in Zs])
        ood[s] = {"n": n, "mean": mu.tolist(), "cov": cov.tolist(),
                  "threshold_95": float(np.quantile(d, 0.95))}
    (C.OUT_DIR / "ood_reference.json").write_text(
        json.dumps(ood, ensure_ascii=False), encoding="utf-8")

    print("k_choice", k_choice)
    print("cum_explained", float(np.sum(pca.explained_variance_ratio_)))
    print("low recon", int((r2 < 0.90).sum()), "/", len(cols))
    print("key recon:")
    print(recon[recon.is_key][["compound", "r2_cv", "rmse_std_cv"]].to_string(index=False))
    print("outer split:")
    print(folds.groupby(["split", "style_fold"]).size().to_string())


if __name__ == "__main__":
    main()
