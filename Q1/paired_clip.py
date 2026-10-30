"""配对检验：ET vs 折内裁剪后的 PLS/Ridge。

输入：
  results/stability_folds.csv         ET 的 30 折 core RMSE
  results/linear_clip_stability.csv   裁剪后 PLS/Ridge 的 30 折 core RMSE
输出：
  results/paired_tests_clip.csv

口径与 stability.py 的 paired_tests.csv 一致：先按 seed 对 3 折取均值，
再做逐 seed 配对 Wilcoxon；delta = 裁剪模型 - ET，delta>0 表示 ET 更好。
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config as C


def main():
    folds = pd.read_csv(C.OUT_DIR / "stability_folds.csv")
    clip = pd.read_csv(C.OUT_DIR / "linear_clip_stability.csv")
    et = folds[folds.model == "et"].groupby("seed").core_rmse.mean()

    rows = []
    for model in ["pls_clip", "ridge_clip"]:
        other = clip[clip.model == model].groupby("seed").core_rmse.mean() \
            .reindex(et.index)
        d = other - et
        try:
            p = wilcoxon(et, other, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        rows.append({
            "comparison": f"et_vs_{model}",
            "mean_delta_rmse": float(d.mean()),
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "wilcoxon_p": float(p),
            "separable": bool((d > 0).sum() >= 8 and d.mean() > 0),
        })

    out = pd.DataFrame(rows)
    out.to_csv(C.OUT_DIR / "paired_tests_clip.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
