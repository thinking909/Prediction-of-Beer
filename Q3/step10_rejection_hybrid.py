"""Hybrid 生成器 + classifier-guided rejection。

- Hybrid：n>=8 用 KDE bw=1.2；n<8 用默认层次 Copula（低置信）。
- Rejection：KDE 生成 5 倍候选，交叉拟合判别器给 fake 概率，
  在低 fake 概率候选里用 farthest-point 选 n_gen 个，保持多样性。
"""
import shutil
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KernelDensity

import q3_config as C
import gen_alt_utils as G

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PY = sys.executable


def farthest_point(Z, k, rng):
    n = len(Z)
    sel = [int(rng.integers(n))]
    d = cdist(Z, Z[sel]).ravel()
    for _ in range(1, k):
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, cdist(Z, Z[[j]]).ravel())
    return sel


def write_variant(df, conc, tag, ctx):
    df = df[["style", "sample_id", "low_confidence"] + ctx.cols].copy()
    df.to_csv(OUT / f"generated_profiles_{tag}.csv", index=False, encoding="utf-8-sig")
    conc = conc[["style", "sample_id", "low_confidence"] + ctx.cols].copy()
    conc.to_csv(OUT / f"generated_concentration_{tag}.csv", index=False, encoding="utf-8-sig")


def evaluate(tag):
    shutil.copy(OUT / f"generated_profiles_{tag}.csv", OUT / "generated_profiles.csv")
    shutil.copy(OUT / f"generated_concentration_{tag}.csv", OUT / "generated_concentration.csv")
    for s in ["step3_constraints.py", "step4_sensory.py", "step6_evaluate.py"]:
        subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    d = pd.read_csv(OUT / "metrics_by_style.csv")
    d.to_csv(OUT / f"{tag}_metrics_by_style.csv", index=False, encoding="utf-8-sig")
    x = d[d.low_confidence == False]
    return {
        "variant": tag, "n_styles": len(x),
        "style_hit_mean": float(x.style_hit_rate.mean()),
        "auc_mean": float(x.real_vs_fake_auc.mean()),
        "auc_median": float(x.real_vs_fake_auc.median()),
        "auc_in_0.4_0.6": int(((x.real_vs_fake_auc >= 0.4) & (x.real_vs_fake_auc <= 0.6)).sum()),
        "mmd_mean": float(x.mmd2.mean()), "coverage_mean": float(x.coverage.mean()),
        "diversity_mean": float(x.diversity_ratio.mean()),
        "corr_dist_mean": float(x.corr_dist.mean()), "ood_mean": float(x.ood_rate.mean()),
    }


def main():
    ctx = G.load_ctx()
    prep = joblib.load(OUT / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    components = np.load(OUT / "pca_model.npz")["components"]

    # ---------- Hybrid ----------
    kde = pd.read_csv(OUT / "generated_profiles_kde_bw1p2.csv", encoding="utf-8-sig")
    kde_c = pd.read_csv(OUT / "generated_concentration_kde_bw1p2.csv", encoding="utf-8-sig")
    cop = pd.read_csv(OUT / "generated_profiles.csv", encoding="utf-8-sig")
    cop_c = pd.read_csv(OUT / "generated_concentration.csv", encoding="utf-8-sig")
    rare = [s for s, n in ctx.style_tr.value_counts().items() if n < C.RARE_THRESHOLD]
    hyb = pd.concat([kde, cop[cop["style"].isin(rare)]], ignore_index=True)
    hyb_c = pd.concat([kde_c, cop_c[cop_c["style"].isin(rare)]], ignore_index=True)
    write_variant(hyb, hyb_c, "hybrid", ctx)
    print("hybrid", hyb.shape, "rare styles", rare, flush=True)

    # ---------- Rejection ----------
    rng = np.random.default_rng(C.SEED)
    frames = []
    for s in sorted(ctx.marg):
        if ctx.n_gen.get(s, 0) <= 0:
            continue
        n_gen = ctx.n_gen[s]
        m = n_gen * 5
        Zs = ctx.Ztr[ctx.style_tr.to_numpy() == s]
        kde_s = KernelDensity(bandwidth=1.2, kernel="gaussian").fit(Zs)
        z = kde_s.sample(m, random_state=int(rng.integers(1e9)))
        Xg = G.finish(z, s, ctx)
        Zg = qt.transform(imp.transform(Xg[ctx.cols])) @ components.T
        Xc = np.vstack([Zs, Zg])
        yc = np.r_[np.zeros(len(Zs)), np.ones(len(Zg))]
        p = cross_val_predict(
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
            method="predict_proba")[:, 1]
        p_c = p[len(Zs):]
        pool = np.argsort(p_c)[: min(len(p_c), n_gen * 3)]
        sel = farthest_point(Zg[pool], n_gen, rng)
        chosen = pool[np.array(sel)]
        Xs = Xg.iloc[chosen].copy()
        Xs["style"] = s
        Xs["sample_id"] = [f"{s}_{i:03d}" for i in range(len(Xs))]
        Xs["low_confidence"] = False
        frames.append(Xs)
        print(f"{s}: {n_gen} selected from {m}, fake_score "
              f"{p_c[chosen].mean():.3f}", flush=True)
    rej = pd.concat(frames, ignore_index=True)
    rej_c = rej.copy()
    for c in ctx.cols:
        if ctx.is_log.get(c, False):
            rej_c[c] = np.power(10.0, rej_c[c])
    write_variant(rej, rej_c, "rejection", ctx)
    print("rejection", rej.shape, flush=True)

    # ---------- Evaluate ----------
    rows = [evaluate("hybrid"), evaluate("rejection")]
    print(pd.DataFrame(rows).to_string(index=False))
    old_f = OUT / "alt_generators_metrics.csv"
    old = pd.read_csv(old_f) if old_f.exists() else pd.DataFrame()
    if not old.empty:
        old = old[~old.variant.isin(["hybrid", "rejection"])]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(
        old_f, index=False, encoding="utf-8-sig")

    # 恢复默认
    for s in ["step2_copula.py", "step3_constraints.py", "step4_sensory.py",
              "step5_transfer.py", "step6_evaluate.py"]:
        subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    print("restored default")


if __name__ == "__main__":
    main()
