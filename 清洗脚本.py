# -*- coding: utf-8 -*-
"""
数据清洗与标签规范化脚本
针对本项目 附件/ 下的 1~8 号文件：
  1) 缺失处理（化学缺失保留原值并另存 0/1 掩码，不做全样本事前填补；全空列删除）
  2) 标签规范化（跨文件统一命名）：
       - 化学物质统一为 human-readable 规范名（来自 8.xlsx 'Sensory thresholds' 表，
         并把希腊字母 mojibake/希腊字符统一为 Alpha/Beta/Gamma/Delta 的 ASCII 写法）
       - 感官属性统一为 CleanName_PanelFeedback（来自 8.xlsx 'Sensory names' 表）
       - 相关矩阵首列 'Unnamed: 0' 规范为 'chemical' / 'sensory_descriptor'
       - 修复表头中的 '\r\n' / '_x000D_' 换行符
  3) 输出到 清洗后/ 目录
"""

import os
import re
import pandas as pd
import numpy as np
import openpyxl
from openpyxl import load_workbook
import shutil
from pathlib import Path

BASE = str(Path(__file__).resolve().parent)
OUT = os.path.join(BASE, "清洗后")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. 读取映射表（来自 8.xlsx）
# ---------------------------------------------------------------
wb = load_workbook(os.path.join(BASE, "8.xlsx"), read_only=True, data_only=True)

th = pd.DataFrame(wb["Sensory thresholds"].iter_rows(values_only=True))
th.columns = th.iloc[0]
th = th.iloc[1:].astype(object)

sn = pd.DataFrame(wb["Sensory names"].iter_rows(values_only=True))
sn.columns = sn.iloc[0]
sn = sn.iloc[1:].astype(object)
wb.close()


def normalize_chem_name(s):
    """把化学物名统一为 ASCII 规范名（希腊字母 mojibake/希腊字符 -> Alpha/Beta/Gamma/Delta）"""
    if s is None:
        return s
    s = str(s)
    # mojibake（UTF-8 双重编码）: Î±(0xCE 0xB1) -> Alpha, Î²(0xCE 0xB2) -> Beta ...
    s = s.replace("Î±", "Alpha")   # Î±
    s = s.replace("Î²", "Beta")    # Î²
    s = s.replace("Î³", "Gamma")   # Î³
    s = s.replace("Î´", "Delta")   # Î´
    s = s.replace("Î’", "Beta")    # Î’ (right single quote 变体)
    # 真正的希腊字符
    s = s.replace("α", "Alpha").replace("β", "Beta")
    s = s.replace("γ", "Gamma").replace("δ", "Delta")
    return s


def normalize_label(s):
    """修复表头中的换行符 / _x000D_ 等 -> 单个空格"""
    if s is None:
        return s
    s = str(s)
    s = s.replace("_x000D_", " ").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# 化学物名映射：field_name_R / field_name_python -> 规范化学名
chem_map_r = {}
chem_map_py = {}
chem_meta = {}  # 规范名 -> (unit, origin_paper, calibrated)
for _, r in th.iterrows():
    cn = normalize_chem_name(r["chemical_name"])
    if pd.isna(cn) or cn == "":
        continue
    fr = r["field_name_R"]
    fp = r["field_name_python"]
    if pd.notna(fr):
        chem_map_r[str(fr)] = cn
    if pd.notna(fp):
        chem_map_py[str(fp)] = cn
    chem_meta.setdefault(cn, (r["unit"], r["origin_paper"], r["calibrated"]))

# 原始 chemical_name（可能含 mojibake）-> 规范名，用于 2.csv 的标签
chem_map_raw = {}
for _, r in th.iterrows():
    cn_raw = r["chemical_name"]
    if pd.isna(cn_raw):
        continue
    chem_map_raw[str(cn_raw)] = normalize_chem_name(cn_raw)

# 感官属性映射（Trained panel）：R_name -> CleanName_PanelFeedback（规范化后）
panel = sn[sn["dataset"] == "Trained panel"].copy()
sens_map = {}
sens_meta = {}
for _, r in panel.iterrows():
    rname = r["R_name"]
    clean = normalize_label(r["CleanName_PanelFeedback"])
    if pd.isna(rname) or pd.isna(clean) or clean == "":
        continue
    sens_map[str(rname)] = clean
    sens_meta[clean] = (r["Category_Paper"], r["Category"], r["DataType"])

ID_COLS = ["beer", "beer_id", "tasting_category_fine"]

log = []  # 处理日志


def logline(msg):
    log.append(msg)
    print(msg)


# ---------------------------------------------------------------
# 2. 处理各文件
# ---------------------------------------------------------------
# ---------- 1.csv 化学测量数据 ----------
d1 = pd.read_csv(os.path.join(BASE, "1.csv"), encoding="utf-8-sig")
orig_cols = list(d1.columns)
# 重命名化学列
rename1 = {}
for c in d1.columns:
    if c in ID_COLS:
        continue
    if c in chem_map_r:
        rename1[c] = chem_map_r[c]
    elif c in chem_map_py:
        rename1[c] = chem_map_py[c]
d1 = d1.rename(columns=rename1)

# 缺失值：保留原始缺失，不做全样本事前填补；另存缺失掩码。
# 原因：全样本填补会把验证折信息带入训练。建模在各训练折内以中位数插补。
chem_cols1 = [c for c in d1.columns if c not in ID_COLS]
na_mask = d1[chem_cols1].isna().astype(int)
na_mask.insert(0, "beer", d1["beer"].to_numpy())
na_mask.to_csv(os.path.join(OUT, "1_化学测量数据_缺失掩码.csv"),
               index=False, encoding="utf-8-sig")
na_cols = [c for c in chem_cols1 if d1[c].isna().any()]
logline(f"[1.csv] 重命名 {len(rename1)} 列；保留原始缺失 "
        f"{int(na_mask[chem_cols1].to_numpy().sum())} 处（{len(na_cols)} 列），另存缺失掩码")
d1.to_csv(os.path.join(OUT, "1_化学测量数据.csv"), index=False, encoding="utf-8-sig")

# 位置审计：原始列名 -> 规范列名（供预处理补充脚本读取）
pos_rows = [{"position": i, "orig_name": o, "clean_name": c}
            for i, (o, c) in enumerate(zip(orig_cols, d1.columns))]
pd.DataFrame(pos_rows).to_csv(os.path.join(OUT, "_position_audit.csv"),
                              index=False, encoding="utf-8-sig")

# ---------- 2.csv 化学参数 Spearman 相关矩阵 ----------
d2 = pd.read_csv(os.path.join(BASE, "2.csv"), encoding="utf-8-sig")
d2 = d2.rename(columns={d2.columns[0]: "chemical"})
d2.columns = [normalize_chem_name(c) if i > 0 else c for i, c in enumerate(d2.columns)]
d2["chemical"] = d2["chemical"].map(lambda s: normalize_chem_name(s))
logline("[2.csv] 首列 Unnamed:0 -> 'chemical'；希腊字母统一为 ASCII（如 α->Alpha, β->Beta）")
d2.to_csv(os.path.join(OUT, "2_化学参数Spearman相关矩阵.csv"), index=False, encoding="utf-8-sig")

# ---------- 3.xlsx 品评表（模板，原样复制） ----------
logline("[3.xlsx] 品评表为空白模板，原样复制")

# ---------- 4.csv 感官分析结果 ----------
d4 = pd.read_csv(os.path.join(BASE, "4.csv"), encoding="utf-8-sig")
rename4 = {}
for c in d4.columns:
    if c in ID_COLS:
        continue
    if c in sens_map:
        rename4[c] = sens_map[c]
d4 = d4.rename(columns=rename4)
logline(f"[4.csv] 重命名 {len(rename4)} 个感官列（缩写 -> CleanName_PanelFeedback 规范名）")
d4.to_csv(os.path.join(OUT, "4_感官分析结果.csv"), index=False, encoding="utf-8-sig")

# ---------- 5.csv 感官参数 Spearman 相关矩阵 ----------
d5 = pd.read_csv(os.path.join(BASE, "5.csv"), encoding="utf-8-sig")
d5 = d5.rename(columns={d5.columns[0]: "sensory_descriptor"})
d5.columns = [normalize_label(c) if i > 0 else c for i, c in enumerate(d5.columns)]
d5["sensory_descriptor"] = d5["sensory_descriptor"].map(normalize_label)
logline("[5.csv] 首列 Unnamed:0 -> 'sensory_descriptor'；修复 'Overall appreciation' 表头换行符")
d5.to_csv(os.path.join(OUT, "5_感官参数Spearman相关矩阵.csv"), index=False, encoding="utf-8-sig")

# ---------- 6.csv 化学-感官 Spearman 相关矩阵 ----------
d6 = pd.read_csv(os.path.join(BASE, "6.csv"), encoding="utf-8-sig")
d6 = d6.rename(columns={d6.columns[0]: "chemical"})
d6.columns = [normalize_label(c) if i > 0 else c for i, c in enumerate(d6.columns)]
d6["chemical"] = d6["chemical"].map(normalize_chem_name)
logline("[6.csv] 首列 Unnamed:0 -> 'chemical'；修复感官列表头换行符")
d6.to_csv(os.path.join(OUT, "6_化学感官参数Spearman相关矩阵.csv"), index=False, encoding="utf-8-sig")

# ---------- 7.csv 论坛评论数据 ----------
d7 = pd.read_csv(os.path.join(BASE, "7.csv"), encoding="utf-8-sig")
allna7 = [c for c in d7.columns if d7[c].isna().all()]
d7 = d7.drop(columns=allna7)
logline(f"[7.csv] 删除 {len(allna7)} 个全空列：{allna7}")
d7.to_csv(os.path.join(OUT, "7_论坛评论数据.csv"), index=False, encoding="utf-8-sig")

# ---------- 8.xlsx 其他相关数据：规范 'Calibrated chemical dataset' 的列名 ----------
src8 = os.path.join(BASE, "8.xlsx")
dst8 = os.path.join(OUT, "8_其他相关数据.xlsx")
wb8 = load_workbook(src8)
if "Calibrated chemical dataset" in wb8.sheetnames:
    ws = wb8["Calibrated chemical dataset"]
    header = [c.value for c in ws[1]]
    new_header = []
    for h in header:
        if h in chem_map_r:
            new_header.append(chem_map_r[h])
        elif h in chem_map_py:
            new_header.append(chem_map_py[h])
        else:
            new_header.append(h)
    for j, val in enumerate(new_header, start=1):
        ws.cell(row=1, column=j, value=val)
    logline("[8.xlsx] 'Calibrated chemical dataset' 工作表化学列名已规范")
wb8.save(dst8)
logline("[8.xlsx] 已另存为 8_其他相关数据.xlsx（其余工作表保持不变）")

# ---------------------------------------------------------------
print("\n===== 全部完成 =====")
print("输出目录：", OUT)
