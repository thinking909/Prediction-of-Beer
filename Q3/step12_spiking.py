"""In-silico spiking 验证（论文加标实验的化学空间版本）。

对 Pils/Lager 源啤酒，把目标风格（Tripel）的指纹 Top-k 化合物调到
目标风格的中位数/95% 分位，其余化合物保持源值；再做严格缺陷封顶。
用 Q1 感官模型、目标风格分类器、源身份和缺陷 OAV 评价。
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import q3_config as C
import gen_alt_utils as G

PROXIES = {
    "Color", "pH", "Protein", "Caloric value", "Carbon dioxide", "Sugars",
    "Unfermentables", "Iron", "Betaglucan", "Ammonia", "Sulfur dioxide",
}


def cap_defects(x, source_x, target_q, cols, is_log):
    out = x.copy()
    for c, kind in [("Dimethyl sulfide", "oav"), ("Diacetyl", "oav"),
                    ("4-vinylguaiacol", "oav_style"), ("E-2-nonenal", "relative")]:
        if c not in cols:
            continue
        sc = float(source_x[c])
        if kind == "oav":
            th = C.DEFECTS[c]["threshold"]
            cap_style = np.log10(th) if is_log.get(c, False) else th
            cap = sc
        elif kind == "oav_style":
            th = C.DEFECTS[c]["threshold"]
            th_chem = np.log10(th) if is_log.get(c, False) else th
            cap = min(sc, min(target_q[c]["q90"], th_chem))
        else:
            cap = min(sc, target_q[c]["q95"])
        out[c] = min(float(out[c]), float(cap))
    return out


def main():
    ctx = G.load_ctx()
    b = ctx.b
    cols = ctx.cols
    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    components = np.load(C.OUT_DIR / "pca_model.npz")["components"]
    sens_model = joblib.load(C.OUT_DIR / "sensory_et.joblib")
    clf = joblib.load(C.OUT_DIR / "style_classifier.joblib")["models"]
    fp = pd.read_csv(C.OUT_DIR / "style_fingerprint.csv", encoding="utf-8-sig")
    meta = pd.read_csv(C.OUT_DIR / "compound_meta.csv", encoding="utf-8-sig")
    is_log = dict(zip(meta.compound, meta.is_log.astype(bool)))

    source, target = "Pils/Lager", "Tripel"
    if target not in clf:
        print("classifier missing", target); return
    targ = ctx.marg[target]
    tq = {}
    for c in cols:
        v = targ[c].dropna()
        tq[c] = {"q50": float(v.quantile(0.5)), "q90": float(v.quantile(0.90)),
                 "q95": float(v.quantile(0.95))}
    fps = list(fp[fp["style"] == target].sort_values("rank").compound)
    fp_set = set(fps)

    non_fp = [c for c in cols if c not in fp_set and c not in PROXIES]
    # 真实目标风格感官均值
    target_idx = ctx.Xtr.index[ctx.style_tr == target]
    real_target_sens = b.sens.loc[target_idx, sens_model["targets"]].mean()
    target_dims = [t for t in ["Esters taste - overall", "Alcohol", "Body",
                               "Hops taste - overall", "Malt taste - overall",
                               "Sweetness", "Bitterness", "Acidity"]
                   if t in sens_model["targets"]]

    # 源样本：外层训练集 Pils/Lager
    src_idx = ctx.Xtr.index[ctx.style_tr == source]
    rows = []
    for k in [5, 10, 15]:
        for level in ["q50", "q95"]:
            for beer in src_idx:
                xs = pd.Series(imp.transform(ctx.Xtr.loc[[beer]])[0], index=cols)
                x = xs.copy()
                for c in fps[:k]:
                    x[c] = tq[c][level]
                x = cap_defects(x, xs, tq, cols, is_log)
                # 风格得分
                Zs = qt.transform(imp.transform(pd.DataFrame([xs[cols]]))) @ components.T
                Zx = qt.transform(imp.transform(pd.DataFrame([x[cols]]))) @ components.T
                p_s = float(clf[target].predict_proba(Zs)[0, 1])
                p_x = float(clf[target].predict_proba(Zx)[0, 1])
                # 感官
                pred = sens_model["model"].predict(
                    sens_model["imputer"].transform(pd.DataFrame([x[cols]])))[0]
                cos = float(np.dot(pred, real_target_sens.to_numpy()) /
                            (np.linalg.norm(pred) * np.linalg.norm(real_target_sens.to_numpy()) + 1e-12))
                pred_d = pd.Series(pred, index=sens_model["targets"])[target_dims]
                real_d = real_target_sens[target_dims]
                cos_d = float(np.dot(pred_d, real_d) /
                              (np.linalg.norm(pred_d) * np.linalg.norm(real_d) + 1e-12))
                # 源身份
                a = xs[non_fp].to_numpy(float)
                bb = x[non_fp].to_numpy(float)
                m = np.isfinite(a) & np.isfinite(bb)
                ident = float(spearmanr(a[m], bb[m]).statistic) if m.sum() > 5 else np.nan
                # 缺陷
                def oav(c, th):
                    return float((10 ** x[c] if is_log.get(c, False) else x[c]) / th)
                rows.append({
                    "source": beer, "target": target, "k": k, "level": level,
                    "p_source": p_s, "p_spiked": p_x,
                    "delta_p": p_x - p_s, "sensory_cosine": cos,
                    "sensory_cosine_target_dims": cos_d,
                    "identity_rho": ident,
                    "dms_oav_source": float((10 ** xs["Dimethyl sulfide"] if is_log.get("Dimethyl sulfide", False) else xs["Dimethyl sulfide"]) / C.DEFECTS["Dimethyl sulfide"]["threshold"]),
                    "dms_oav_spiked": oav("Dimethyl sulfide", C.DEFECTS["Dimethyl sulfide"]["threshold"]),
                    "diacetyl_oav_source": float((10 ** xs["Diacetyl"] if is_log.get("Diacetyl", False) else xs["Diacetyl"]) / C.DEFECTS["Diacetyl"]["threshold"]),
                    "diacetyl_oav_spiked": oav("Diacetyl", C.DEFECTS["Diacetyl"]["threshold"]),
                    "4vg_oav_source": float((10 ** xs["4-vinylguaiacol"] if is_log.get("4-vinylguaiacol", False) else xs["4-vinylguaiacol"]) / C.DEFECTS["4-vinylguaiacol"]["threshold"]),
                    "4vg_oav_spiked": oav("4-vinylguaiacol", C.DEFECTS["4-vinylguaiacol"]["threshold"]),
                    "t2n_source": float(xs["E-2-nonenal"]),
                    "t2n_spiked": float(x["E-2-nonenal"]),
                })
    d = pd.DataFrame(rows)
    d.to_csv(C.OUT_DIR / "in_silico_spiking_cases.csv", index=False, encoding="utf-8-sig")
    g = d.groupby(["k", "level"]).agg(
        n=("source", "size"), p_source=("p_source", "mean"),
        p_spiked=("p_spiked", "mean"), delta_p=("delta_p", "mean"),
        sensory_cosine=("sensory_cosine", "mean"),
        sensory_cosine_target_dims=("sensory_cosine_target_dims", "mean"),
        identity_rho=("identity_rho", "mean"),
        dms_source=("dms_oav_source", "mean"), dms_spiked=("dms_oav_spiked", "mean"),
        dia_source=("diacetyl_oav_source", "mean"), dia_spiked=("diacetyl_oav_spiked", "mean"),
        vg_source=("4vg_oav_source", "mean"), vg_spiked=("4vg_oav_spiked", "mean"),
        t2n_source=("t2n_source", "mean"), t2n_spiked=("t2n_spiked", "mean"),
    ).reset_index()
    g.to_csv(C.OUT_DIR / "in_silico_spiking_summary.csv", index=False, encoding="utf-8-sig")
    print(g.to_string(index=False))


if __name__ == "__main__":
    main()
