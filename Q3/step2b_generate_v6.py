"""Q3 Step 2b：v6 生成器（高斯秩变换 + 风格条件收缩高斯 + 平滑逆变换）。

与 step2_copula 的关系：
- 同样在训练集上做高斯秩变换、在 70 个绝对列上建模；
- 采样改为：风格相关矩阵向全局收缩（n0=8），在 GRT 空间按 sqrt((n+1)/(n-1)) 做方差补偿；
- 逆变换用逐列经验分位（平滑线性插值），不做事后线性方差重标定；
- 稀有风格（n<RARE_THRESHOLD）沿用族级/全局边缘，标低置信。

输出（与现有流程同构）：
  generated_profiles.csv / generated_profiles_v6.csv
  generated_concentration.csv
  generation_meta.csv / generation_summary.csv / generation_params.json
"""
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

import q3_config as C
import gen_alt_utils as G

DEFECTS = ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"]
EPS = 1e-3
N0 = 8.0
N_MAIN, N_OTHER = 100, 50


def shrink_spd(R, Rg, lam):
    R = (1 - lam) * 0.5 * (R + R.T) + lam * Rg
    w, V = np.linalg.eigh(0.5 * (R + R.T))
    w = np.clip(w, EPS, None)
    R2 = (V * w) @ V.T
    d = np.sqrt(np.diag(R2))
    return R2 / np.outer(d, d)


def main():
    C.ensure_dirs()
    ctx = G.load_ctx()
    b = ctx.b
    cols = ctx.cols
    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]

    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    trm = folds.split == "outer_train"
    Xtr = b.chem[cols].loc[folds.loc[trm, "beer"].to_numpy()]
    style_tr = folds.loc[trm, "style_orig"].set_axis(Xtr.index)
    Ximp = pd.DataFrame(imp.transform(Xtr), columns=cols, index=Xtr.index)
    Ztr = qt.transform(Ximp)
    Rg = np.nan_to_num(np.corrcoef(Ztr, rowvar=False))
    Rg = 0.5 * (Rg + Rg.T)
    rng = np.random.default_rng(C.SEED)
    is_log = ctx.is_log

    prof_rows, conc_rows, meta, summary = [], [], [], []
    for s in sorted(set(folds.style_orig)):
        idx = np.where(style_tr.to_numpy() == s)[0]
        n = len(idx)
        n_gen = N_MAIN if s in C.MAIN_STYLES else N_OTHER
        low = bool(n < C.RARE_THRESHOLD)
        lam = N0 / (N0 + n) if n >= 3 else 1.0
        Rh = (np.nan_to_num(np.corrcoef(Ztr[idx], rowvar=False))
              if n >= 3 else Rg)
        R = shrink_spd(Rh, Rg, lam)
        w, V = np.linalg.eigh(R)
        z = rng.normal(size=(n_gen, len(cols))) @ (V * np.sqrt(np.clip(w, 0, None))).T
        if n >= 3:
            z = z * np.sqrt((n + 1) / (n - 1))
        P = norm.cdf(z)
        use_style = n >= 3
        src = Ximp.loc[style_tr.index[idx], cols] if use_style else ctx.marg[s][cols]
        Xg = pd.DataFrame(index=range(n_gen), columns=cols, dtype=float)
        for j, c in enumerate(cols):
            v = src[c].dropna().to_numpy(float)
            Xg[c] = np.quantile(v, P[:, j], method="linear") if len(v) >= 3 else np.nan
        for c in DEFECTS:
            v = src[c].dropna()
            if len(v) and c in Xg:
                Xg[c] = np.minimum(Xg[c].to_numpy(), float(v.quantile(0.95)))

        Xc = Xg.copy()
        for c in cols:
            if is_log.get(c, False):
                Xc[c] = np.power(10.0, Xc[c])
        for i in range(n_gen):
            row = Xg.iloc[i].to_dict()
            row.update({"style": s, "sample_id": f"{s}_{i:03d}", "low_confidence": low})
            prof_rows.append(row)
            rowc = Xc.iloc[i].to_dict()
            rowc.update({"style": s, "sample_id": f"{s}_{i:03d}", "low_confidence": low})
            conc_rows.append(rowc)
        meta.append({"style": s, "n_train": n, "n_gen": n_gen,
                     "low_confidence": low, "lambda_cov": lam,
                     "marginal_source": "style" if use_style else "family/global",
                     "variance_comp": "sqrt((n+1)/(n-1))" if n >= 3 else "none"})
        summary.append({"style": s, "n_train": n, "n_gen": n_gen, "low_confidence": low})

    gen = pd.DataFrame(prof_rows)
    order = cols + ["style", "sample_id", "low_confidence"]
    gen = gen[order]
    gen.to_csv(C.OUT_DIR / "generated_profiles.csv", index=False, encoding="utf-8-sig")
    gen.to_csv(C.OUT_DIR / "generated_profiles_v6.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conc_rows)[order].to_csv(C.OUT_DIR / "generated_concentration.csv",
                                          index=False, encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(C.OUT_DIR / "generation_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(C.OUT_DIR / "generation_summary.csv",
                                 index=False, encoding="utf-8-sig")
    (C.OUT_DIR / "generation_params.json").write_text(json.dumps({
        "generator": "v6: GRT + style-conditional shrunk Gaussian + smooth quantile inverse",
        "n0_shrink": N0, "eig_floor": EPS,
        "variance_compensation": "rank-space sqrt((n+1)/(n-1))",
        "marginal": "empirical per style (rare: family/global), linear interpolation",
        "no_variance_matching": True, "defects": "capped at style/family 95th",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved generated_profiles.csv:", gen.shape,
          "| styles", gen["style"].nunique(),
          "| low_conf", int(gen.low_confidence.sum()))


if __name__ == "__main__":
    main()
