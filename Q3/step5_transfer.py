"""Q3 Step 5：跨风格迁移（T0/T1/T3）+ 风格分类器 + 迁移指标。

案例：Pils/Lager→Tripel、Blond→Tripel、Kriek→Lambic。
方法：
  T0 均值平移；
  T1 正则化 Gaussian OT；
  T3 OT + 源感知缺陷约束（主推）。

评价：目标风格得分、源身份保持（非指纹化合物）、缺陷 Δ、感官余弦。
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

import q3_config as C
import dataio

PROXIES = {
    "Color", "pH", "Protein", "Caloric value", "Carbon dioxide", "Sugars",
    "Unfermentables", "Iron", "Betaglucan", "Ammonia", "Sulfur dioxide",
}


def sqrtm_psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


def invsqrtm_psd(A):
    w, V = np.linalg.eigh((A + A.T) / 2)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def moments(Z, style, target, mask, n0=C.N0_SHRINK):
    global_mu = Z.mean(axis=0)
    global_cov = np.cov(Z, rowvar=False)
    sel = (style == target) & mask
    zs = Z[sel]
    mu = zs.mean(axis=0)
    n = len(zs)
    lam = n0 / (n0 + n)
    cov = (1 - lam) * np.cov(zs, rowvar=False) + lam * global_cov
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    return mu, cov, n


def cap_defects(x, source_x, style, cols, is_log, target_q, mode="strict"):
    out = x.copy()
    for c, kind in [("Dimethyl sulfide", "oav"), ("Diacetyl", "oav"),
                    ("4-vinylguaiacol", "oav_style"), ("E-2-nonenal", "relative")]:
        if c not in cols:
            continue
        sc = float(source_x[c])
        if kind == "oav":
            th = C.DEFECTS[c]["threshold"]
            cap_style = np.log10(th) if is_log.get(c, False) else th
            cap = sc if mode == "strict" else max(sc, cap_style)
        elif kind == "oav_style":
            th = C.DEFECTS[c]["threshold"]
            th_chem = np.log10(th) if is_log.get(c, False) else th
            style_cap = min(target_q[c]["q90"], th_chem)
            cap = min(sc, style_cap) if mode == "strict" else max(sc, style_cap)
        else:  # t2n
            style95 = target_q[c]["q95"]
            iqr = target_q[c]["q75"] - target_q[c]["q25"]
            tol = 0.05 * iqr
            cap = min(sc, style95) if mode == "strict" else min(sc + tol, style95)
        out[c] = min(float(out[c]), float(cap))
    return out


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    X = b.chem[cols].copy()
    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    trm = folds.split == "outer_train"
    Xtr = X.loc[folds.loc[trm, "beer"].to_numpy()]
    style_tr = folds.loc[trm, "style_orig"].set_axis(Xtr.index)
    style_fold_tr = folds.loc[trm, "style_fold"].set_axis(Xtr.index)
    inner = folds.loc[trm, "inner_fold"].set_axis(Xtr.index)

    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    pca_npz = np.load(C.OUT_DIR / "pca_model.npz")
    components = pca_npz["components"]

    Zall = qt.transform(imp.transform(X)) @ components.T
    Ztr = Zall[X.index.get_indexer(Xtr.index)]
    meta = pd.read_csv(C.OUT_DIR / "compound_meta.csv", encoding="utf-8-sig")
    is_log = dict(zip(meta.compound, meta.is_log.astype(bool)))
    fp = pd.read_csv(C.OUT_DIR / "style_fingerprint.csv", encoding="utf-8-sig")

    # 目标风格的训练集分位数（chem scale）
    def target_quantiles(style):
        sub = Xtr.loc[style_tr == style]
        out = {}
        for c in cols:
            v = sub[c].dropna()
            out[c] = {"q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75)),
                      "q90": float(v.quantile(0.90)), "q95": float(v.quantile(0.95))}
        return out

    # 目标风格 one-vs-rest 分类器（PCA 得分）
    test = folds[~trm]
    Xte = X.loc[test.beer.to_numpy()]
    Zte = qt.transform(imp.transform(Xte)) @ components.T
    targets = sorted({t for _, t in C.TRANSFER_CASES})
    clfs, clf_metrics = {}, []
    for target in targets:
        ytr = (style_tr == target).astype(int).to_numpy()
        yte = (test.style_orig == target).astype(int).to_numpy()
        clf = LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=C.SEED)
        clf.fit(Ztr, ytr)
        p = clf.predict(Zte)
        auc = roc_auc_score(yte, clf.predict_proba(Zte)[:, 1]) if yte.min() < yte.max() else float("nan")
        clf_metrics.append({
            "target": target, "n_pos_train": int(ytr.sum()), "n_pos_test": int(yte.sum()),
            "balanced_accuracy": float(balanced_accuracy_score(yte, p)),
            "auc": float(auc),
        })
        clfs[target] = clf
    joblib.dump({"models": clfs}, C.OUT_DIR / "style_classifier.joblib")
    pd.DataFrame(clf_metrics).to_csv(C.OUT_DIR / "style_classifier_metrics.csv",
                                     index=False, encoding="utf-8-sig")

    def p_target(Z, target):
        if target not in clfs:
            return np.full(len(Z), np.nan)
        return clfs[target].predict_proba(Z)[:, 1]

    # 迁移
    rows, sens_rows, metric_rows = [], [], []
    for source, target in C.TRANSFER_CASES:
        if source not in style_tr.unique() or target not in style_tr.unique():
            continue
        tq = target_quantiles(target)
        fp_target = set(fp[fp["style"] == target].compound)
        non_fp = [c for c in cols if c not in fp_target and c not in PROXIES]

        for k in sorted(inner.unique()):
            src_idx = style_tr[(style_tr == source) & (inner == k)].index
            if len(src_idx) == 0:
                continue
            mask = inner != k
            muS, covS, nS = moments(Ztr, style_tr, source, mask)
            muT, covT, nT = moments(Ztr, style_tr, target, mask)
            A = sqrtm_psd(covT) @ invsqrtm_psd(covS)
            for beer in src_idx:
                zs = Zall[X.index.get_loc(beer)]
                xsource = X.loc[beer, cols].to_numpy()
                for method, zt in [
                    ("T0", zs + muT - muS),
                    ("T1", muT + A @ (zs - muS)),
                ]:
                    zt = np.asarray(zt).ravel()
                    xhat = qt.inverse_transform((zt @ components).reshape(1, -1))[0]
                    xhat = pd.Series(np.clip(xhat, Xtr[cols].quantile(0.01).to_numpy(),
                                             Xtr[cols].quantile(0.99).to_numpy()), index=cols)
                    if method == "T1":
                        # T3: T1 + source-aware defect caps
                        src_series = pd.Series(xsource, index=cols)
                        x3s = cap_defects(xhat, src_series, target, cols, is_log, tq, "strict")
                        x3t = cap_defects(xhat, src_series, target, cols, is_log, tq, "threshold")
                        for name, xx in [("T1", xhat), ("T3", x3s), ("T3t", x3t)]:
                            xx = pd.Series(np.clip(xx.to_numpy(),
                                                   Xtr[cols].quantile(0.01).to_numpy(),
                                                   Xtr[cols].quantile(0.99).to_numpy()),
                                           index=cols)
                            rec = {"case": f"{source}->{target}", "split": f"inner{k}",
                                   "method": name, "source": beer,
                                   "source_style": source, "target_style": target}
                            rec.update({c: float(xx[c]) for c in cols})
                            rows.append(rec)
                    else:
                        rec = {"case": f"{source}->{target}", "split": f"inner{k}",
                               "method": method, "source": beer,
                               "source_style": source, "target_style": target}
                        rec.update({c: float(xhat[c]) for c in cols})
                        rows.append(rec)

    tr = pd.DataFrame(rows)
    if len(tr) == 0:
        print("no transfer rows")
        return
    tr.to_csv(C.OUT_DIR / "transfer_profiles.csv", index=False, encoding="utf-8-sig")

    # 还原浓度
    trc = tr.copy()
    for c in cols:
        if is_log.get(c, False):
            trc[c] = np.power(10.0, trc[c])
    trc.to_csv(C.OUT_DIR / "transfer_concentration.csv", index=False, encoding="utf-8-sig")

    # 感官预测
    sm = joblib.load(C.OUT_DIR / "sensory_et.joblib")
    pred = sm["model"].predict(sm["imputer"].transform(tr[cols]))
    pred = pd.DataFrame(pred, columns=sm["targets"])
    pred.insert(0, "source", tr["source"].to_numpy())
    pred.insert(0, "method", tr["method"].to_numpy())
    pred.insert(0, "case", tr["case"].to_numpy())
    pred.to_csv(C.OUT_DIR / "transfer_sensory.csv", index=False, encoding="utf-8-sig")

    # 源感官
    src_pred = sm["model"].predict(sm["imputer"].transform(
        X.loc[tr["source"].unique(), cols]))
    src_pred = pd.DataFrame(src_pred, columns=sm["targets"], index=tr["source"].unique())

    # 指标
    for case in tr.case.unique():
        source, target = case.split("->")
        real_t = b.sens.loc[Xtr.index[style_tr == target], sm["targets"]].mean()
        for method in ["T0", "T1", "T3", "T3t"]:
            sub = tr[(tr.case == case) & (tr.method == method)]
            if len(sub) == 0:
                continue
            # 目标风格得分
            zt = qt.transform(imp.transform(sub[cols])) @ components.T
            p_t = p_target(zt, target)
            zs = Zall[X.index.get_indexer(sub["source"].to_numpy())]
            p_s = p_target(zs, target)
            # 源身份
            fp_target = set(fp[fp["style"] == target].compound)
            non_fp = [c for c in cols if c not in fp_target and c not in PROXIES]
            ids = []
            for i, (_, r) in enumerate(sub.iterrows()):
                a = X.loc[r["source"], non_fp].to_numpy(float)
                b2 = r[non_fp].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(b2)
                ids.append(float(spearmanr(a[m], b2[m]).statistic) if m.sum() > 5 else np.nan)
            # 感官余弦
            pp = pred[(pred.case == case) & (pred.method == method)][sm["targets"]].mean().to_numpy()
            rr = real_t.to_numpy()
            cos = float(np.dot(pp, rr) / (np.linalg.norm(pp) * np.linalg.norm(rr) + 1e-12))
            # 缺陷
            subc = trc[(trc.case == case) & (trc.method == method)]
            dms = C.DEFECTS["Dimethyl sulfide"]["threshold"]
            dia = C.DEFECTS["Diacetyl"]["threshold"]
            vg = C.DEFECTS["4-vinylguaiacol"]["threshold"]
            src_c = b.chem_restored.loc[sub["source"].to_numpy(), ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"]]
            metric_rows.append({
                "case": case, "method": method, "n": len(sub),
                "p_target_source": float(np.mean(p_s)),
                "p_target_trans": float(np.mean(p_t)),
                "delta_p_target": float(np.mean(p_t) - np.mean(p_s)),
                "identity_rho": float(np.mean(ids)),
                "sensory_cosine_target": cos,
                "dms_oav_source": float(np.median(src_c["Dimethyl sulfide"] / dms)),
                "dms_oav_trans": float(np.median(subc["Dimethyl sulfide"] / dms)),
                "diacetyl_oav_source": float(np.median(src_c["Diacetyl"] / dia)),
                "diacetyl_oav_trans": float(np.median(subc["Diacetyl"] / dia)),
                "4vg_oav_source": float(np.median(src_c["4-vinylguaiacol"] / vg)),
                "4vg_oav_trans": float(np.median(subc["4-vinylguaiacol"] / vg)),
                "t2n_source": float(np.median(src_c["E-2-nonenal"])),
                "t2n_trans": float(np.median(subc["E-2-nonenal"])),
            })
    pd.DataFrame(metric_rows).to_csv(C.OUT_DIR / "transfer_metrics.csv",
                                     index=False, encoding="utf-8-sig")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
