"""Acetaldehyde 化学代理等级。

4.csv 没有 acetaldehyde 感官标签，无法训练感官模型；但 3.xlsx 单卡有该行。
这里用 1.csv 的绝对浓度生成一个**化学代理** 0–6 等级（浓度分位），
并在 fault_flags 中同时保留 OAV（阈值 50 mg/L，全部低于阈值）。

明确：这不是品评组评分，只是为了让评分卡不缺项的代理分。

输出：
  results/acetaldehyde_proxy.csv
"""
import numpy as np
import pandas as pd

import config as C
import dataio

TARGET = "Acetaldehyde"
THRESHOLD = 50.0  # mg/L，来自 8.xlsx Sensory thresholds


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    conc = b.chem_restored[TARGET].sort_index()
    logc = np.log10(conc.to_numpy())
    cuts = np.quantile(logc, np.arange(1, C.N_LEVELS) / C.N_LEVELS)
    level = np.clip(np.searchsorted(cuts, logc, side="right"), 0, C.N_LEVELS - 1)

    out = pd.DataFrame({
        "beer": conc.index,
        "concentration_mgL": conc.to_numpy(),
        "oav": conc.to_numpy() / THRESHOLD,
        "proxy_level": level,
    }).set_index("beer")
    out["oav_flag"] = np.where(out.oav > 1, "above", "below")
    out.to_csv(C.OUT_DIR / "acetaldehyde_proxy.csv")
    print("level counts", np.bincount(level, minlength=7).tolist(),
          "| OAV max", round(float(out.oav.max()), 3))


if __name__ == "__main__":
    main()
