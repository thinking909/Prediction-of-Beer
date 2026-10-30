"""复核 2.csv 化学相关矩阵：重算 Pearson/Spearman 与附件矩阵的平均绝对差。

输出：
  附件/清洗后/相关矩阵复核.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "附件"
CLEAN = BASE / "清洗后"

raw = pd.read_csv(BASE / "1.csv", encoding="utf-8-sig")
scale = pd.read_csv(CLEAN / "化学字段单位与尺度表.csv", encoding="utf-8-sig")
rename = dict(zip(scale["原始名"], scale["列名"]))
raw = raw.rename(columns=rename)
cols = list(scale["列名"])
X = raw.set_index("beer").sort_index()[cols].apply(pd.to_numeric, errors="coerce")

# 用清洗后的附件矩阵（231x231，无缺失），与论文“26565 对”口径一致
att = pd.read_csv(CLEAN / "2_化学参数Spearman相关矩阵.csv",
                  encoding="utf-8-sig", index_col=0)
att = att.reindex(index=cols, columns=cols)
iu = np.triu_indices(len(cols), k=1)

res = {"n_cols": len(cols), "n_pairs": int(len(iu[0]))}
for name in ["pearson", "spearman"]:
    m = X.corr(method=name).to_numpy()[iu]
    a = att.to_numpy()[iu]
    ok = np.isfinite(m) & np.isfinite(a)
    res[f"{name}_mean_abs_diff"] = float(np.mean(np.abs(m[ok] - a[ok])))
    res[f"{name}_pairs"] = int(ok.sum())

(CLEAN / "相关矩阵复核.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=2))
