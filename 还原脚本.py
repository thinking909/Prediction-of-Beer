# -*- coding: utf-8 -*-
"""
尺度还原脚本：基于证据层级判定 1_化学测量数据.csv 各列的量纲
证据层级（由强到弱）：
  E1 精确匹配 8.xlsx Calibrated chemical dataset（斜率=0.4343 → log10；残差=0 → raw）
  E2 含负值 → log10（原始浓度非负；z-score 假设已被分布统计否定）
  E3 派生加和列：含负值 → log10(总和)；构成另行核对
  E4 无法判定 → 保留原值并标注 unresolved（附偏度/锚定证据）
输出：
  清洗后/1_化学测量数据_还原.csv     （log10 列 10^x 还原；raw 列保持；unresolved 保持+标注）
  清洗后/变换审计_全231列.csv        （逐列判定与证据）
  清洗后/还原报告.json               （汇总统计）
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import skew
from pathlib import Path

BASE = str(Path(__file__).resolve().parent)
CLEAN = BASE + "/清洗后"

MOJI = {"Î±": "Alpha", "Î²": "Beta", "Î³": "Gamma", "Î´": "Delta",
        "α": "Alpha", "β": "Beta", "γ": "Gamma", "δ": "Delta"}


def norm(s):
    s = str(s)
    for k, v in MOJI.items():
        s = s.replace(k, v)
    return s


d1 = pd.read_csv(CLEAN + "/1_化学测量数据.csv", encoding="utf-8-sig")
orig = pd.read_csv(BASE + "/1.csv", encoding="utf-8-sig")
th = pd.read_excel(CLEAN + "/8_其他相关数据.xlsx", sheet_name="Sensory thresholds")
cal = pd.read_excel(CLEAN + "/8_其他相关数据.xlsx", sheet_name="Calibrated chemical dataset").set_index("beer")

chem_map = {}
for _, r in th.iterrows():
    cn = norm(r["chemical_name"])
    if pd.notna(r["field_name_R"]):
        chem_map[str(r["field_name_R"])] = cn
    if pd.notna(r["field_name_python"]):
        chem_map[str(r["field_name_python"])] = cn
orig2clean = {}
for c in orig.columns:
    if c in ("beer", "beer_id", "tasting_category_fine"):
        continue
    orig2clean[c] = norm(chem_map.get(c, c))

d1i = d1.set_index("beer")
origi = orig.set_index("beer")
orig_by_clean = {}
for c in orig.columns:
    if c in ("beer", "beer_id", "tasting_category_fine"):
        continue
    orig_by_clean[norm(chem_map.get(c, c))] = origi[c]
SUM_COLS = {"Total hop aroma compounds", "Total acids", "Total sulfur compounds", "Total ester compounds"}
rows, restored = {}, {}
audit = []
for c in d1.columns:
    if c in ("beer", "beer_id", "tasting_category_fine"):
        continue
    x = pd.to_numeric(d1i[c], errors="coerce")
    xf = x.dropna().to_numpy(float)
    has_neg = bool((xf < 0).any())
    evidence, verdict = "", ""
    if c in cal.columns:
        v = pd.to_numeric(cal.loc[d1i.index, c], errors="coerce")
        m = x.notna() & v.notna() & (v > 0)
        if m.sum() >= 40:
            xv, vv = x[m].to_numpy(float), v[m].to_numpy(float)
            with np.errstate(over="ignore"):
                e10 = float(np.median(np.abs(np.power(10.0, xv) - vv) / np.maximum(vv, 1e-12))) if np.max(np.abs(xv)) < 30 else np.inf
            eid = float(np.median(np.abs(xv - vv) / np.maximum(np.abs(vv), 1e-12)))
            if e10 < 1e-3:
                verdict, evidence = "log10", f"E1: 10^x 与校准集中位相对误差 {e10:.1e}"
            elif eid < 1e-3:
                verdict, evidence = "raw", f"E1: 与校准集完全一致(相对误差 {eid:.1e})"
            else:
                verdict, evidence = "unresolved", "E1f: 与校准集相关但非精确匹配"
    elif c in SUM_COLS:
        if has_neg:
            verdict, evidence = "log10", "E3: 派生加和列且含负值 → log10(总和)"
        else:
            verdict, evidence = "unresolved_sum", "E3f: 派生加和列、无负值、成员构成未确证"
    elif has_neg:
        verdict, evidence = "log10", "E2: 含负值(原始浓度非负)且非z-score"
    elif c in SUM_COLS:
        verdict, evidence = "unresolved_sum", "E3f: 派生加和列、无负值、成员构成未确证"
    else:
        verdict, evidence = "semi_quant", ("E4: 未校准HS-SPME-GC-MS相对峰面积(内标4-fluorobenzaldehyde归一化, 任意单位); "
                                          "论文Methods: 峰面积积分+内标归一化, 未做校准曲线; "
                                          "因通过Shapiro近正态检验而未做log变换 -> 无需还原, 保持原值; "
                                          "仅相对半定量可用, 不参与阈值(mg/L)比较")
    if verdict == "log10":
        restored[c] = x.map(lambda t: np.nan if pd.isna(t) else float(np.power(10.0, t)))
    else:
        restored[c] = x.copy()
    if c in orig_by_clean:
        o = pd.to_numeric(orig_by_clean[c], errors="coerce").reindex(d1i.index)
        imputed = int(o.isna().sum())
    else:
        imputed = 0
    audit.append({
        "列名": c, "判定": verdict, "证据": evidence,
        "是否含负值": has_neg, "偏度": round(float(skew(xf)), 3) if len(xf) > 8 else None,
        "还原后范围": f"{restored[c].min():.3g} ~ {restored[c].max():.3g}" if pd.notna(restored[c]).any() else "",
        "原始缺失数": imputed,
    })

ad = pd.DataFrame(audit)
ad.to_csv(CLEAN + "/变换审计_全231列.csv", index=False, encoding="utf-8-sig")

out = d1.copy()
for c in d1.columns:
    if c in ("beer", "beer_id", "tasting_category_fine"):
        continue
    out[c] = np.asarray(restored[c], dtype=float)
out.to_csv(CLEAN + "/1_化学测量数据_还原.csv", index=False, encoding="utf-8-sig", float_format="%.10g")

# 还原质量门: 37个精确列 10^x 应与校准集一致
errs = []
outb = out.set_index("beer")
for c in ad[(ad["判定"] == "log10") & (ad["证据"].str.startswith("E1"))]["列名"]:
    y = pd.to_numeric(outb[c], errors="coerce")
    v = pd.to_numeric(cal[c], errors="coerce").reindex(y.index)
    m = y.notna() & v.notna() & (v > 0)
    errs.append(float(np.median(np.abs(y[m] - v[m]) / np.maximum(v[m], 1e-12))))
gate = float(np.max(errs))

report = {
    "总列数": int(len(ad)),
    "判定分布": ad["判定"].value_counts().to_dict(),
    "E1门检(37列10^x vs 校准集最大中位相对误差)": gate,
    "log10列数(还原为10^x)": int((ad["判定"] == "log10").sum()),
    "raw列数(保持原值)": int((ad["判定"] == "raw").sum()),
    "半定量/未决列(保持原值)": ad[ad["判定"].str.startswith(("unresolved", "semi_quant"))]["列名"].tolist(),
    "半定量列说明": "15列为未校准HS-SPME-GC-MS相对峰面积(内标归一化, 任意单位), 本就未做log变换, 无需还原; "
                    "不参与阈值比较; 1列(Total hop aroma)为派生加和且构成未确证",
    "校准浓度说明": "仅70个'Calibrated chemical dataset'列有绝对浓度; 其余列的10^x还原值为相对量/未校准单位, 论文引用时需注明",
    "缺失说明": "清洗表保留原始缺失(NaN), 不做全样本事前填补; 建模在训练折内以中位数插补",
}
with open(CLEAN + "/还原报告.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 派生加和列构成核查
hop_members = ["Linalool", "Geraniol", "Citronellol", "Alpha-terpineol", "Caryophyllene",
              "Alpha-humulene", "Z-nerolidol", "E-Beta-ocimene", "E-Beta-damascenone",
              "E,E-2,4-decadienal", "Eugenol", "Gamma-nonalactone", "E-2-nonenal", "Diethyl succinate", "Ethyl nicotinate"]
if "Total hop aroma compounds" in out.columns:
    mem = [c for c in hop_members if c in out.columns]
    s = out[mem].sum(axis=1)
    t = d1i["Total hop aroma compounds"]
    m = np.isfinite(s.to_numpy(float)) & np.isfinite(t.to_numpy(float)) & (s.to_numpy(float) > 0)
    r_log = float(np.corrcoef(np.log10(s.to_numpy(float)[m]), t.to_numpy(float)[m])[0, 1])
    report["酒花香气加和核查_r(log10成员和, 派生列)"] = round(r_log, 4)
    with open(CLEAN + "/还原报告.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps({k: v for k, v in report.items() if k != "未决列偏度证据"}, ensure_ascii=False, indent=2))
