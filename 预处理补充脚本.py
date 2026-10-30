# -*- coding: utf-8 -*-
"""
预处理补充三项:
  P0-1 化学字段单位与绝对/相对标志表 (231 列)
  P0-2 派生列核查: 反向求解 Total acids/ester/sulfur/hop aroma/Caloric 的成员与系数
  P0-3 异常值系统审计: 稳健 z (median/MAD) 逐列扫描 + 明细
输出统一到 清洗后/:
  化学字段单位与尺度表.csv
  派生列核查报告.csv
  异常值审计_逐列.csv
  异常值明细_top50.csv
  预处理补充报告.json
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from pathlib import Path

BASE = str(Path(__file__).resolve().parent / "清洗后")
d1 = pd.read_csv(BASE + "/1_化学测量数据.csv", encoding="utf-8-sig").set_index("beer")
rest = pd.read_csv(BASE + "/1_化学测量数据_还原.csv", encoding="utf-8-sig").set_index("beer")
ad = pd.read_csv(BASE + "/变换审计_全231列.csv", encoding="utf-8-sig")
pos = pd.read_csv(BASE + "/_position_audit.csv", encoding="utf-8-sig")
th = pd.read_excel(BASE + "/8_其他相关数据.xlsx", sheet_name="Sensory thresholds")
cal = pd.read_excel(BASE + "/8_其他相关数据.xlsx", sheet_name="Calibrated chemical dataset")
cal_cols = set(cal.columns) - {"beer", "beer_id", "tasting_category_fine"}
MOJI = {"Î±": "Alpha", "Î²": "Beta", "Î³": "Gamma", "Î´": "Delta", "α": "Alpha", "β": "Beta", "γ": "Gamma", "δ": "Delta"}


def norm(s):
    s = str(s)
    for k, v in MOJI.items():
        s = s.replace(k, v)
    return s


th["cn"] = th["chemical_name"].map(norm)
thm = th.drop_duplicates("cn").set_index("cn")
CLASS = dict(zip(ad["列名"], ad["判定"]))
orig_name = dict(zip(pos["clean_name"], pos["orig_name"]))


def unit_from_field(field):
    if pd.isna(field):
        return ""
    f = str(field)
    for pat, u in [(".mg.L.", "mg/L"), (".mg.l.", "mg/L"), (".g.L.", "g/L"), (".g.l.", "g/L"),
                   ("..v.v.", "% v/v"), (".EBC.", "EBC"), ("IBU", "IBU"), ("kcalperc", "kcal/100mL"),
                   (".mg.kg.", "mg/kg")]:
        if pat in f:
            return u
    return ""


rows = []
for c in [x for x in d1.columns if x not in ("beer_id", "tasting_category_fine")]:
    meta = thm.loc[c] if c in thm.index else None
    unit = ""
    if meta is not None:
        if pd.notna(meta.get("unit")):
            unit = str(meta["unit"])
        else:
            unit = unit_from_field(meta.get("field_name_R"))
    cls = CLASS.get(c, "")
    is_cal = c in cal_cols
    if c in ("Total acids", "Total ester compounds", "Total sulfur compounds", "Total hop aroma compounds"):
        kind, absrel, thr_ok = "派生加和", "派生", "否"
    elif cls == "semi_quant":
        kind, absrel, thr_ok = "相对峰面积(内标归一化)", "相对", "否"
    elif is_cal:
        kind, absrel = ("log10还原" if cls == "log10" else "raw(未变换)"), "绝对"
        thr_ok = "是" if (meta is not None and pd.notna(meta.get("threshold_mean"))) else "有阈值表可查"
        if meta is not None and pd.notna(meta.get("threshold_mean")):
            thr_ok = "是"
        else:
            thr_ok = "阈值缺失"
    else:
        kind, absrel, thr_ok = ("未校准(log10还原)" if cls == "log10" else "未校准"), "相对", "否"
    rows.append({
        "列名": c,
        "原始名": orig_name.get(c, ""),
        "尺度类别": kind,
        "绝对/相对": absrel,
        "单位": unit,
        "是否在校准集": "是" if is_cal else "否",
        "阈值均值": float(meta["threshold_mean"]) if (meta is not None and pd.notna(meta.get("threshold_mean"))) else None,
        "阈值中位": float(meta["threshold_median"]) if (meta is not None and pd.notna(meta.get("threshold_median"))) else None,
        "来源(origin_paper)": str(meta["origin_paper"]) if (meta is not None and pd.notna(meta.get("origin_paper"))) else "",
        "可做阈值判定": thr_ok,
    })
u = pd.DataFrame(rows)
u.to_csv(BASE + "/化学字段单位与尺度表.csv", index=False, encoding="utf-8-sig")
print("=== P0-1 单位表 ===")
print("总列数:", len(u), "| 绝对/相对分布:", u["绝对/相对"].value_counts().to_dict())
print("绝对列中有单位标注:", int(((u["绝对/相对"] == "绝对") & (u["单位"] != "")).sum()), "/", int((u["绝对/相对"] == "绝对").sum()))
print("绝对列中可做阈值判定:", int((u["可做阈值判定"] == "是").sum()))

# ---------------- P0-2 派生列核查 ----------------
DERIVED = ["Total acids", "Total ester compounds", "Total sulfur compounds",
           "Total hop aroma compounds", "Caloric value"]
abs_cols = [c for c in u[u["绝对/相对"] == "绝对"]["列名"] if c in rest.columns and c not in DERIVED]
A_all = rest[abs_cols].to_numpy(float)
A_all = np.clip(A_all, 0, None)


def greedy_fit(y, A, cols, max_k=8, tol=5e-4):
    sel, best_rel, coefs = [], np.inf, None
    for _ in range(max_k):
        cand = None
        for j in range(A.shape[1]):
            if j in sel:
                continue
            S = sel + [j]
            c, res = nnls(A[:, S], y)
            rel = res / max(np.linalg.norm(y), 1e-12)
            if cand is None or rel < cand[0]:
                cand = (rel, j, S, c)
        if cand[0] < tol:
            sel, best_rel, coefs = cand[2], cand[0], cand[3]
            break
        if sel and cand[0] > best_rel * 0.995:
            break
        sel, best_rel, coefs = cand[2], cand[0], cand[3]
    return sel, best_rel, coefs


der_rows = []
for d in DERIVED:
    if d not in rest.columns:
        continue
    raw = rest[d].to_numpy(float)
    y = np.power(10.0, d1[d].to_numpy(float)) if CLASS.get(d) == "log10" else raw
    y = np.clip(y, 0, None)
    row_ok = np.isfinite(A_all).all(axis=1) & np.isfinite(y)
    sel, rel, coefs = greedy_fit(y[row_ok], A_all[row_ok], abs_cols)
    members = ";".join(f"{abs_cols[j]}×{coefs[i]:.4g}" for i, j in enumerate(sel))
    der_rows.append({"派生列": d, "预期类型": CLASS.get(d), "10^x值域": f"{np.nanmin(y):.4g}~{np.nanmax(y):.4g}",
                     "相对残差": round(float(rel), 6), "识别成员与系数": members,
                     "判定": "成员可复原(残差<0.5%)" if rel < 5e-3 else ("部分可复原" if rel < 0.1 else "无法复原(成员缺失或尺度不符)")})
der = pd.DataFrame(der_rows)
der.to_csv(BASE + "/派生列核查报告.csv", index=False, encoding="utf-8-sig")
print("\n=== P0-2 派生列核查 ===")
print(der[["派生列", "相对残差", "判定"]].to_string(index=False))
for _, r in der.iterrows():
    print(f"  {r['派生列']}: {r['识别成员与系数']}")

# ---------------- P0-3 异常值审计 ----------------
rel = d1[[c for c in d1.columns if c not in ("beer_id", "tasting_category_fine")]].to_numpy(float)
cols = [c for c in d1.columns if c not in ("beer_id", "tasting_category_fine")]
per_col, detail = [], []
for j, c in enumerate(cols):
    x = rel[:, j]
    q1, q3 = np.nanpercentile(x, [25, 75])
    iqr = q3 - q1
    hi = q3 + 3 * iqr
    idx_far = np.where(x > hi)[0] if iqr > 0 else np.array([], dtype=int)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad if mad > 0 else (iqr / 1.349 if iqr > 0 else np.nan)
    xv = x[np.isfinite(x)]
    vals, counts = np.unique(xv, return_counts=True)
    mode_share = float(counts.max() / len(xv))
    rest_med = float(np.nanmedian(rest[c].to_numpy(float)))
    rest_max = float(np.nanmax(rest[c].to_numpy(float)))
    ratio = rest_max / rest_med if rest_med > 0 else np.nan
    z = (x - med) / scale if np.isfinite(scale) and scale > 0 else np.full_like(x, np.nan)
    per_col.append({"列名": c, "尺度类别": CLASS.get(c, ""), "绝对值": "是" if c in cal_cols else "否",
                    "n_Tukey远超": int(len(idx_far)), "n_|z|>10": int(np.nansum(np.abs(z) > 10)),
                    "最大|z|": round(float(np.nanmax(np.abs(z))), 2) if np.isfinite(scale) and scale > 0 else None,
                    "异常beer": int(d1.index[int(np.nanargmax(np.abs(z)))]) if np.isfinite(scale) and scale > 0 else "",
                    "还原后max/median": round(ratio, 3) if np.isfinite(ratio) else None,
                    "最大取值占比": round(mode_share, 3)})
    for k in idx_far:
        detail.append({"列名": c, "beer": int(d1.index[k]), "释放尺度值": round(float(rel[k, j]), 4),
                       "还原值": float(rest.iloc[k][c]), "还原max/median": round(float(rest.iloc[k][c]) / rest_med, 2) if rest_med > 0 else None,
                       "稳健z": round(float(z[k]), 1) if np.isfinite(z[k]) else None,
                       "尺度类别": CLASS.get(c, ""), "是否校准列": "是" if c in cal_cols else "否"})
pc = pd.DataFrame(per_col).sort_values("n_Tukey远超", ascending=False)
pc.to_csv(BASE + "/异常值审计_逐列.csv", index=False, encoding="utf-8-sig")
dt_all = pd.DataFrame(detail)
if len(dt_all):
    by_beer = dt_all.groupby("beer").agg(远超单元数=("列名", "size"), 涉及列=("列名", lambda s: ", ".join(sorted(set(s))[:6]))).reset_index()
    by_beer = by_beer.sort_values("远超单元数", ascending=False)
    by_beer.to_csv(BASE + "/异常值_按beer汇总.csv", index=False, encoding="utf-8-sig")
    print("\n异常最集中的样本(前8):")
    print(by_beer.head(8).to_string(index=False))
    dt = dt_all.assign(absratio=dt_all["还原max/median"].abs()).sort_values(
        ["absratio", "稳健z"], ascending=False).head(50).drop(columns="absratio")
else:
    dt = dt_all
dt.to_csv(BASE + "/异常值明细_top50.csv", index=False, encoding="utf-8-sig")
print("\n=== P0-3 异常值审计(Tukey远超 Q3+3IQR + 稳健z 分级) ===")
print(f"远超单元总数: {int(pc['n_Tukey远超'].sum())} | 存在远超的列: {int((pc['n_Tukey远超']>0).sum())}/{len(pc)}")
print(pc[pc["n_Tukey远超"] > 0][["列名", "n_Tukey远超", "最大|z|", "异常beer", "还原后max/median", "最大取值占比"]].head(12).to_string(index=False))
print("\n可信极端(还原max/median>100):")
print(dt[dt["还原max/median"] > 100][["列名", "beer", "还原值", "还原max/median", "尺度类别"]].head(15).to_string(index=False))
print("\n还原后绝对列负值数(应为0):",
      int((rest[[c for c in u[u['绝对/相对'] == '绝对']['列名']]] < 0).sum().sum()))

DEFECT = {"Dimethyl sulfide": "dms", "Diacetyl": "diacetyl", "4-vinylguaiacol": "X4vg", "E-2-nonenal": "t2n"}
def_rows = []
for chem, sens in DEFECT.items():
    if chem not in rest.columns or chem not in thm.index:
        continue
    m = thm.loc[chem]
    thr = m.get("threshold_mean")
    if pd.isna(thr):
        continue
    v = rest[chem].to_numpy(float)
    vv = v[np.isfinite(v)]
    n_over = int((vv > float(thr)).sum())
    def_rows.append({"化学列": chem, "感官指标": sens, "单位": str(m.get("unit", "")),
                     "阈值均值": float(thr), "超阈样本数": n_over,
                     "超阈比例": f"{n_over / max(len(vv), 1):.1%}",
                     "浓度范围": f"{np.nanmin(vv):.3g}~{np.nanmax(vv):.3g}"})
if def_rows:
    dd = pd.DataFrame(def_rows)
    dd.to_csv(BASE + "/缺陷化合物超阈统计.csv", index=False, encoding="utf-8-sig")
    print("\n=== 附: 缺陷化合物超阈统计(校准列, 可与阈值比较) ===")
    print(dd.to_string(index=False))

rep = {
    "P0-1 单位表": {"总列数": len(u), "绝对/相对": u["绝对/相对"].value_counts().to_dict(),
                    "绝对列有单位": int(((u["绝对/相对"] == "绝对") & (u["单位"] != "")).sum()),
                    "可做阈值判定列数": int((u["可做阈值判定"] == "是").sum())},
    "P0-2 派生列": {r["派生列"]: {"相对残差": r["相对残差"], "判定": r["判定"]} for _, r in der.iterrows()},
    "P0-3 异常": {"远超单元数": int(pc["n_Tukey远超"].sum()),
                  "存在远超的列数": int((pc["n_Tukey远超"] > 0).sum()),
                  "top列": pc[pc["n_Tukey远超"] > 0].head(5)[["列名", "n_Tukey远超", "最大|z|"]].to_dict("records")},
}
with open(BASE + "/预处理补充报告.json", "w", encoding="utf-8") as f:
    json.dump(rep, f, ensure_ascii=False, indent=2)
print("\n报告已写出: 预处理补充报告.json")
