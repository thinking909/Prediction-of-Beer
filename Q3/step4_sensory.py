"""Q3 Step 4：感官画像读出、OOD 检测、风格指纹校验。

  - 在 V3 70 列上训练 ET（46 个主模型维），只用外层训练集；
  - 用 5 折 OOF 计算逐维 R²，标注弱维；
  - 预测生成谱的感官画像；
  - 用共享 PCA 空间 + 风格 Mahalanobis 距离做 OOD 标记；
  - 检查每个风格的指纹化合物是否落在真实 5–95% 区间。

输出：
  results/sensory_model_metrics.csv
  results/sensory_profiles_generated.csv
  results/sensory_profiles_strict.csv
  results/sensory_style_summary.csv
  results/ood_flags.csv
  results/fingerprint_check.csv
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors

import q3_config as C
import dataio


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    targets = list(C.q1c.MODEL_TARGETS)
    X = b.chem[cols].copy()
    Y = b.sens[targets].copy()
    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    tr = folds[folds.split == "outer_train"].copy()
    Xtr = X.loc[tr.beer.to_numpy()]
    Ytr = Y.loc[tr.beer.to_numpy()]
    inner = tr.set_index("beer")["inner_fold"].reindex(Xtr.index)

    # ---- 5 折 OOF 逐维 R² ----
    oof = np.full(Ytr.shape, np.nan)
    for k in sorted(inner.unique()):
        itr = inner.index[inner != k]
        iva = inner.index[inner == k]
        imp = SimpleImputer(strategy="median").fit(Xtr.loc[itr])
        et = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 4})
        et.fit(imp.transform(Xtr.loc[itr]), Ytr.loc[itr])
        oof[inner == k] = et.predict(imp.transform(Xtr.loc[iva]))
    metrics = []
    for j, t in enumerate(targets):
        r2 = r2_score(Ytr[t].to_numpy(), oof[:, j])
        metrics.append({"target": t, "r2_oof": float(r2), "weak": bool(r2 < 0.10)})
    met = pd.DataFrame(metrics).sort_values("r2_oof", ascending=False)
    met.to_csv(C.OUT_DIR / "sensory_model_metrics.csv", index=False, encoding="utf-8-sig")

    # ---- 最终模型 ----
    imp = SimpleImputer(strategy="median").fit(Xtr)
    et = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 4})
    et.fit(imp.transform(Xtr), Ytr)
    joblib.dump({"imputer": imp, "model": et, "targets": targets},
                C.OUT_DIR / "sensory_et.joblib")

    for name, path in [("generated", "constrained_profiles.csv"),
                       ("strict", "strict_threshold_profiles.csv")]:
        gen = pd.read_csv(C.OUT_DIR / path, encoding="utf-8-sig")
        pred = et.predict(imp.transform(gen[cols]))
        pred = pd.DataFrame(pred, columns=targets)
        pred.insert(0, "style", gen["style"].to_numpy())
        pred.insert(1, "sample_id", gen["sample_id"].to_numpy())
        pred.to_csv(C.OUT_DIR / f"sensory_profiles_{name}.csv",
                    index=False, encoding="utf-8-sig")

    pred = pd.read_csv(C.OUT_DIR / "sensory_profiles_generated.csv", encoding="utf-8-sig")
    real_mean = Ytr.groupby(tr.set_index("beer")["style_orig"].reindex(Ytr.index)).mean()
    gen_mean = pred.groupby("style")[targets].mean()
    strong = set(met[~met.weak].target)
    strong_cols = [t for t in targets if t in strong]
    rows = []
    for s in gen_mean.index:
        if s not in real_mean.index:
            continue
        a = gen_mean.loc[s, targets].to_numpy()
        bb = real_mean.loc[s, targets].to_numpy()
        cos = float(np.dot(a, bb) / (np.linalg.norm(a) * np.linalg.norm(bb) + 1e-12))
        mae = float(np.mean(np.abs(a - bb)))
        a2 = gen_mean.loc[s, strong_cols].to_numpy()
        b2 = real_mean.loc[s, strong_cols].to_numpy()
        cos2 = float(np.dot(a2, b2) / (np.linalg.norm(a2) * np.linalg.norm(b2) + 1e-12))
        rows.append({"style": s, "n_gen": int((pred["style"] == s).sum()),
                     "cosine_all": cos, "mae_all": mae,
                     "cosine_strong": cos2, "n_strong": len(strong_cols)})
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "sensory_style_summary.csv",
                              index=False, encoding="utf-8-sig")
    sens_rows = rows

    # ---- OOD：kNN 距离（比 Mahalanobis 更稳健） ----
    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    pca_npz = np.load(C.OUT_DIR / "pca_model.npz")
    components = pca_npz["components"]
    gen = pd.read_csv(C.OUT_DIR / "constrained_profiles.csv", encoding="utf-8-sig")
    Ztr = prep["quantile"].transform(prep["imputer"].transform(Xtr)) @ components.T
    Zgen = prep["quantile"].transform(prep["imputer"].transform(gen[cols])) @ components.T
    style_tr = tr.set_index("beer")["style_orig"].reindex(Xtr.index)
    flags = []
    for s, idx in gen.groupby("style").groups.items():
        real_pos = np.where(style_tr.to_numpy() == s)[0]
        gen_pos = gen.index.get_indexer(idx)
        if len(real_pos) < 3:
            for i in gen_pos:
                flags.append({"style": s, "sample_id": gen.loc[i, "sample_id"],
                              "knn_dist": np.nan, "threshold_95": np.nan, "ood": "na"})
            continue
        Zr = Ztr[real_pos]
        nn2 = NearestNeighbors(n_neighbors=2).fit(Zr)
        dr, _ = nn2.kneighbors(Zr)
        thr = float(np.quantile(dr[:, 1], 0.95))
        nn1 = NearestNeighbors(n_neighbors=1).fit(Zr)
        dg, _ = nn1.kneighbors(Zgen[gen_pos])
        for i, d in zip(gen_pos, dg[:, 0]):
            flags.append({"style": s, "sample_id": gen.loc[i, "sample_id"],
                          "knn_dist": float(d), "threshold_95": thr,
                          "ood": bool(d > thr)})
    pd.DataFrame(flags).to_csv(C.OUT_DIR / "ood_flags.csv",
                               index=False, encoding="utf-8-sig")

    # ---- 风格指纹校验（用外层训练集的 5–95% 区间） ----
    fp = pd.read_csv(C.OUT_DIR / "style_fingerprint.csv", encoding="utf-8-sig")
    rows = []
    for s in gen["style"].unique():
        fs = fp[fp["style"] == s]
        if len(fs) == 0:
            continue
        g = gen[gen["style"] == s]
        ratios = []
        for c in fs.compound:
            vals = Xtr.loc[style_tr == s, c].dropna()
            if len(vals) < 3:
                continue
            lo, hi = np.quantile(vals, [0.05, 0.95])
            ratios.append(float(np.mean((g[c] >= lo) & (g[c] <= hi))))
        if ratios:
            rows.append({"style": s, "n_fp": len(ratios),
                         "mean_in_interval": float(np.mean(ratios)),
                         "min_in_interval": float(np.min(ratios))})
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "fingerprint_check.csv",
                              index=False, encoding="utf-8-sig")

    print("sensory OOF top:")
    print(met.head(8).to_string(index=False))
    print("weak dims", int(met.weak.sum()), "/", len(met))
    print("sensory style summary:")
    print(pd.DataFrame(sens_rows).to_string(index=False) if sens_rows else "no rows")
    print("OOD rate:")
    o = pd.read_csv(C.OUT_DIR / "ood_flags.csv", encoding="utf-8-sig")
    o["ood_bool"] = o["ood"].map({"True": True, "False": False, "na": np.nan})
    print(o.groupby("style").ood_bool.mean().sort_values(ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
