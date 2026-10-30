"""Q3 Step 2（B1）：共享 PCA + 层次 Gaussian Copula + 经验边缘。

做法：
  - 复用 step1 的 imputer / quantile / PCA，在 PCA 得分空间估计每风格均值/协方差；
  - 向族级/全局收缩，采样 z；
  - 把 z 还原成 70 维正态得分，转成 uniform，再用**每风格经验分位数**
    映射回化学尺度（Gaussian Copula + empirical marginals）。

这样避免在偏态/零膨胀化合物上直接加高斯残差导致极端值（Diacetyl/t2n）。

输出：
  results/generated_profiles.csv（化学尺度：log 列仍为 log10）
  results/generated_concentration.csv（还原浓度，供 OAV 使用）
  results/generation_meta.csv
  results/generation_summary.csv
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import q3_config as C
import dataio


def shrink_cov(cov_s, cov_prior, n, n0):
    lam = n0 / (n0 + n)
    cov = (1 - lam) * cov_s + lam * cov_prior
    return cov + np.eye(cov.shape[0]) * 1e-6, lam


def empirical_quantile(values, u):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.full(len(u), np.nan)
    return np.quantile(v, np.clip(u, 0.001, 0.999), method="linear")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--var-match", dest="var_match", action="store_true", default=True)
    ap.add_argument("--no-var-match", dest="var_match", action="store_false")
    args = ap.parse_args()
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    X = b.chem[cols].copy()
    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    train_mask = folds.split == "outer_train"
    Xtr = X.loc[folds.loc[train_mask, "beer"].to_numpy()]
    style_tr = folds.loc[train_mask, "style_orig"].set_axis(Xtr.index)

    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    pca_npz = np.load(C.OUT_DIR / "pca_model.npz")
    components = pca_npz["components"]
    k = int(pca_npz["k"])

    Ztr = qt.transform(imp.transform(Xtr))
    Ximp = pd.DataFrame(imp.transform(Xtr), columns=cols, index=Xtr.index)
    Zhat_tr = Ztr @ components.T @ components
    mean_hat = Zhat_tr.mean(axis=0)
    sd_hat = Zhat_tr.std(axis=0)
    sd_hat[sd_hat < 1e-8] = 1.0

    scores = Ztr @ components.T
    global_mean = scores.mean(axis=0)
    global_cov = np.cov(scores, rowvar=False)

    sour_styles = [s for s in C.SOUR_FAMILY if s in style_tr.unique()]
    if len(sour_styles):
        sour_mask = style_tr.isin(sour_styles).to_numpy()
        sour_mean = scores[sour_mask].mean(axis=0)
        sour_cov = np.cov(scores[sour_mask], rowvar=False)
        sour_vals = Ximp.loc[style_tr.isin(sour_styles)]
    else:
        sour_mean, sour_cov, sour_vals = global_mean, global_cov, Ximp

    meta = pd.read_csv(C.OUT_DIR / "compound_meta.csv", encoding="utf-8-sig")
    is_log = dict(zip(meta.compound, meta.is_log.astype(bool)))

    rng = np.random.default_rng(C.SEED)
    gen_rows, meta_rows, summary = [], [], []

    for style in sorted(style_tr.unique()):
        idx = style_tr[style_tr == style].index
        n = len(idx)
        zs = scores[Xtr.index.get_indexer(idx)]
        mu = zs.mean(axis=0)
        cov_s = np.cov(zs, rowvar=False) if n > 1 else global_cov

        if style in C.SOUR_FAMILY and len(sour_styles):
            mu_prior, cov_prior, marginal_src = sour_mean, sour_cov, sour_vals
        else:
            mu_prior, cov_prior, marginal_src = global_mean, global_cov, Ximp

        # n>=8 用本风格均值；稀有风格才向族级/全局收缩
        lam_mean = 0.0 if n >= C.RARE_THRESHOLD else C.N0_SHRINK / (C.N0_SHRINK + n)
        mu_use = (1 - lam_mean) * mu + lam_mean * mu_prior
        cov_use, lam_cov = shrink_cov(cov_s, cov_prior, n, C.N0_SHRINK)

        # 经验边缘：n>=8 用本风格；稀有风格用族级/全局
        if n >= C.RARE_THRESHOLD:
            marg = Ximp.loc[idx]
            marg_src = "style"
        else:
            marg = marginal_src
            marg_src = "family" if style in C.SOUR_FAMILY else "global"

        n_gen = C.N_GEN_MAIN if style in C.MAIN_STYLES else C.N_GEN_OTHER
        z_gen = rng.multivariate_normal(mu_use, cov_use, size=n_gen)
        Z_hat = z_gen @ components
        Z_std = (Z_hat - mean_hat) / sd_hat
        # 用生成样本的秩做 uniform，保证每个边缘的分布与风格经验分位数一致
        U = np.column_stack([
            rankdata(Z_std[:, j]) / (n_gen + 1) for j in range(Z_std.shape[1])
        ])

        # 对边缘做 winsorize，避免极值；缺陷化合物额外封顶在风格 95% 分位
        lo_q = args.lo if args.lo is not None else (0.05 if n < 15 else 0.025)
        hi_q = args.hi if args.hi is not None else (0.95 if n < 15 else 0.975)
        marg_clip = marg.copy()
        for c in cols:
            lo = marg_clip[c].quantile(lo_q)
            hi = marg_clip[c].quantile(hi_q)
            marg_clip[c] = marg_clip[c].clip(lo, hi)
        X_gen = np.zeros_like(U)
        for j, c in enumerate(cols):
            X_gen[:, j] = empirical_quantile(marg_clip[c].to_numpy(), U[:, j])
        X_gen = pd.DataFrame(X_gen, columns=cols)
        if args.var_match:
            for c in cols:
                if c in ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"]:
                    continue
                mu = float(marg[c].mean())
                sd = float(marg[c].std())
                sdg = float(X_gen[c].std())
                if sd > 0 and sdg > 1e-12:
                    X_gen[c] = mu + (X_gen[c] - mu) * (sd / sdg)
                    X_gen[c] = X_gen[c].clip(float(marg[c].min()), float(marg[c].max()))
        for c in ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"]:
            if c in cols:
                cap = float(marg[c].quantile(0.95))
                X_gen[c] = np.minimum(X_gen[c].to_numpy(), cap)

        low_conf = bool(n < C.RARE_THRESHOLD)
        for i in range(n_gen):
            row = X_gen.iloc[i].to_dict()
            row["style"] = style
            row["sample_id"] = f"{style}_{i:03d}"
            row["low_confidence"] = low_conf
            gen_rows.append(row)

        Xc = X_gen.copy()
        for c in cols:
            if is_log.get(c, False):
                Xc[c] = np.power(10.0, Xc[c])
        for i in range(n_gen):
            row = Xc.iloc[i].to_dict()
            row["style"] = style
            row["sample_id"] = f"{style}_{i:03d}"
            row["low_confidence"] = low_conf
            meta_rows.append(row)

        summary.append({
            "style": style, "n_train": n, "n_gen": n_gen,
            "low_confidence": low_conf,
            "lambda_mean": lam_mean, "lambda_cov": lam_cov,
            "family_prior": "sour" if style in C.SOUR_FAMILY else "global",
            "marginal_source": marg_src,
        })
        print(f"{style:<22} n={n:<3} gen={n_gen:<3} low={low_conf} "
              f"lam_cov={lam_cov:.2f} marg={marg_src}", flush=True)

    gen = pd.DataFrame(gen_rows)
    gen.to_csv(C.OUT_DIR / "generated_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta_rows).to_csv(C.OUT_DIR / "generated_concentration.csv",
                                   index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(C.OUT_DIR / "generation_meta.csv", index=False, encoding="utf-8-sig")

    real_mean = Ximp.groupby(style_tr).mean()
    gen_mean = gen.groupby("style")[cols].mean()
    rows = []
    for s in gen_mean.index:
        if s not in real_mean.index:
            continue
        for c in C.KEY_COMPOUNDS:
            if c in cols:
                rows.append({"style": s, "compound": c,
                             "real_mean": float(real_mean.loc[s, c]),
                             "gen_mean": float(gen_mean.loc[s, c]),
                             "diff": float(gen_mean.loc[s, c] - real_mean.loc[s, c])})
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "generation_summary.csv",
                              index=False, encoding="utf-8-sig")

    (C.OUT_DIR / "generation_params.json").write_text(json.dumps({
        "k": k,
        "n0": C.N0_SHRINK,
        "marginal": "empirical per style (rare: family/global); winsorized 1-99% (5-95% if n<15); defects capped at style 95th",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved generated_profiles.csv:", gen.shape)


if __name__ == "__main__":
    main()
