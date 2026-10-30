"""4 个未进入主模型的维度：统一单维 ET 回退。

目标：DMS、Diacetyl、4-VG、t2n（config.THRESHOLD_TARGETS）。
同一特征（V2）、同一 ET 配置、同一 3 折 OOF（seed 20261023）、
同一 leave-fold-out 等张校准与 1/7…6/7 分位阈值。

这些分数只用于 3.xlsx 的 0–6 交付；OAV 绝对标志另见 fault_threshold_flags.csv。
输出：
  results/fallback_levels_4.csv
  results/fallback_metrics.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

import config as C
import dataio
import models

SEED = C.STABILITY_SEEDS[0]
LEVELS = np.arange(1, C.N_LEVELS) / C.N_LEVELS


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    X = b.X("V2_scale_aware")
    folds = b.outer_folds(SEED)

    levels = pd.DataFrame(index=b.beer)
    metrics = []
    for tgt in C.THRESHOLD_TARGETS:
        y = b.sens[tgt].to_numpy()
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
            et = models.final_et()
            et.fit(imp.transform(X.iloc[tr]), y[tr])
            oof[te] = et.predict(imp.transform(X.iloc[te]))

        cal = np.full(len(y), np.nan)
        for tr, te in folds:
            other = np.setdiff1d(np.arange(len(y)), te)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof[other], y[other])
            cal[te] = iso.predict(oof[te])

        thr = np.quantile(y, LEVELS)
        lv = np.clip(np.searchsorted(thr, cal, side="right"), 0, C.N_LEVELS - 1)
        levels[f"level|{tgt}"] = lv
        levels[f"calibrated|{tgt}"] = cal
        metrics.append({
            "target": tgt,
            "r2_oof": r2_score(y, oof),
            "spearman_oof": spearmanr(oof, y).statistic,
            "level_counts": np.bincount(lv, minlength=C.N_LEVELS).tolist(),
        })

    levels.index.name = "beer"
    levels.to_csv(C.OUT_DIR / "fallback_levels_4.csv")
    m = pd.DataFrame(metrics)
    m.to_csv(C.OUT_DIR / "fallback_metrics.csv", index=False)
    print(m[["target", "r2_oof", "spearman_oof"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
