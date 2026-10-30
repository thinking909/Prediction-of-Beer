"""把 50 维相对 0–6 等级填入 3.xlsx（品评表）的副本。

口径（2026-10 修订）：
- 46 个主模型维：`relative_scores_v3.csv` 的 quantile 等级
- 4 个未进入主模型的维（DMS/Diacetyl/4-VG/t2n）：`fallback_levels_4.csv` 的单维 ET 回退等级
- OAV 绝对标志不再混进 0–6，单独写入 `fault_flags` 工作表
- 50 维全部是模型相对等级

输出：
  附件/3_品评表_相对等级.xlsx（新增 predictions_250 / fault_flags / 说明）
  results/levels_short_50.csv
"""
import numpy as np
import pandas as pd
from openpyxl import load_workbook

import config as C
import dataio

ID_COLS = {"beer", "beer_id", "tasting_category_fine"}

# 12 个品评卡 sheet 中的打分单元格（Aroma 在 Q 列，Flavour 在 AG 列）
SHEET_CELLS = {
    "A_malt_all": ("Q", 7), "A_malt_grain": ("Q", 10),
    "A_malt_bread": ("Q", 12), "A_malt_cara": ("Q", 14),
    "A_malt_burn": ("Q", 16),
    "A_hops_all": ("Q", 20), "A_hops_citrus": ("Q", 23),
    "A_hops_tropical": ("Q", 25), "A_hops_noble": ("Q", 27),
    "A_hops_woody": ("Q", 29),
    "A_esters_all": ("Q", 33), "A_esters_ethac": ("Q", 36),
    "A_esters_isoaa": ("Q", 38), "A_esters_flower": ("Q", 40),
    "A_esters_fruity": ("Q", 42),
    "F_malt_all": ("AG", 7), "F_malt_grain": ("AG", 10),
    "F_malt_bread": ("AG", 12), "F_malt_cara": ("AG", 14),
    "F_malt_burn": ("AG", 16),
    "F_hops_all": ("AG", 20), "F_hops_citrus": ("AG", 23),
    "F_hops_tropical": ("AG", 25), "F_hops_noble": ("AG", 27),
    "F_hops_woody": ("AG", 29),
    "F_esters_all": ("AG", 33), "F_esters_ethac": ("AG", 36),
    "F_esters_isoaa": ("AG", 38), "F_esters_flower": ("AG", 40),
    "F_esters_fruity": ("AG", 42),
    "body": ("Q", 50), "alcohol": ("AG", 50),
    "bitternes": ("Q", 54), "sweetness": ("AG", 54),
    "co2": ("Q", 58), "acidity": ("AG", 58),
    "aftertaste": ("Q", 62),
    "dms": ("Q", 68), "t2n": ("AG", 68),
    "metallic": ("Q", 72), "diacetyl": ("AG", 72),
    "X4vg": ("Q", 76), "stale_hops": ("AG", 76),
    "orange": ("Q", 82), "lactic": ("AG", 82),
    "coriander": ("Q", 86), "acetic": ("AG", 86),
    "clove": ("Q", 90), "barnyard": ("AG", 90),
    "overall": ("Q", 94),
}
SUMMARY_ALIAS = {"A_ester_fruity": "A_esters_fruity",
                 "F_ester_fruity": "F_esters_fruity",
                 "4vg": "X4vg"}

# 品评卡中不属于 50 维官方标签的单元格；每次运行都显式列出，禁止静默留空
CARD_EXCEPTIONS = {
    "Q45": "acetaldehyde 化学代理（无感官标签）",
    "AG45": "acetaldehyde 化学代理（无感官标签）",
    "AG62": "无对应标签的重复 aftertaste 格，留空",
    "AG94": "Comments 文本格，留空",
}


def main():
    C.ensure_dirs()
    raw = pd.read_csv(C.ROOT / "附件" / "4.csv").set_index(C.ID_COL)
    clean = pd.read_csv(C.SENS_FILE).set_index(C.ID_COL)
    orig_cols = [c for c in raw.columns if c not in ID_COLS]
    clean_cols = [c for c in clean.columns if c not in ID_COLS]
    assert len(orig_cols) == len(clean_cols) == 50
    short = dict(zip(clean_cols, orig_cols))          # 长名 -> 4.csv 原始短名
    assert short["Malt aroma - overall"] == "A_malt_all"
    assert short["Dimethyl sulfide"] == "dms"
    assert short["4-vinyl guaiacol"] == "X4vg"
    assert short["Bitterness"] == "bitternes"

    # 46 个主模型维
    lv = pd.read_csv(C.OUT_DIR / "relative_scores_v3.csv", index_col=0)
    level_cols = {c.split("|", 1)[1]: c for c in lv.columns
                  if c.startswith("level|")}
    table = pd.DataFrame(index=lv.index)
    for long_name, col in level_cols.items():
        table[short[long_name]] = lv[col].astype(int)
    modeled = set(table.columns)

    # 4 个回退维
    fb = pd.read_csv(C.OUT_DIR / "fallback_levels_4.csv", index_col=0)
    fallback = set()
    for long_name in C.THRESHOLD_TARGETS:
        s = short[long_name]
        table[s] = fb[f"level|{long_name}"].reindex(table.index).astype(int)
        fallback.add(s)

    # 路径完整性：50 维 = 46 主模型 + 4 回退，不重不漏
    assert modeled | fallback == set(orig_cols), \
        set(orig_cols) ^ (modeled | fallback)
    assert not (modeled & fallback)
    assert table.notna().all().all()

    proxy = pd.read_csv(C.OUT_DIR / "acetaldehyde_proxy.csv", index_col=0)

    table = table[orig_cols]
    out = pd.concat([raw[["tasting_category_fine"]], table], axis=1).sort_index()
    out.index.name = "beer"
    out.to_csv(C.OUT_DIR / "levels_short_50.csv")

    # 先审计模板：Q/AG 列的公式打分格必须全部有归属
    tpl = load_workbook(C.ROOT / "附件" / "3.xlsx")
    known_cells = {f"{c}{r}" for c, r in SHEET_CELLS.values()}
    for n in range(1, 13):
        ws_t = tpl[f"beer_{n}"]
        found = {c.coordinate for col in ("Q", "AG") for c in ws_t[col]
                 if c.data_type == "f"}
        extra = found - known_cells
        if extra != set(CARD_EXCEPTIONS):
            raise ValueError(f"beer_{n} 未登记的评分格: {sorted(extra)}")
    print("card exceptions:", "; ".join(f"{k}={v}" for k, v in CARD_EXCEPTIONS.items()))

    wb = load_workbook(C.ROOT / "附件" / "3.xlsx")
    for name in ("predictions_250", "fault_flags", "说明"):
        if name in wb.sheetnames:
            del wb[name]

    ws = wb.create_sheet("predictions_250")
    ws.append(["Beer", "tasting_category_fine", *orig_cols, "acetaldehyde_proxy"])
    for beer, row in out.iterrows():
        ws.append([int(beer), row.tasting_category_fine,
                   *[int(row[c]) for c in orig_cols],
                   int(proxy.loc[beer, "proxy_level"])])
    ws.freeze_panes = "C2"

    # OAV 绝对标志（独立于 0–6 等级）
    ff = pd.read_csv(C.OUT_DIR / "fault_threshold_flags.csv")
    add = pd.DataFrame({
        "beer": proxy.index,
        "chemical": "Acetaldehyde",
        "sensory_target": "(无感官标签)",
        "threshold": 50.0,
        "unit": "mg/L",
        "concentration": proxy["concentration_mgL"].to_numpy(),
        "oav": proxy["oav"].to_numpy(),
        "flag": proxy["oav_flag"].to_numpy(),
        "judgeable": True,
    })
    ff = pd.concat([ff, add[ff.columns]], ignore_index=True)
    ws = wb.create_sheet("fault_flags")
    ws.append(list(ff.columns))
    for row in ff.itertuples(index=False):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else v
                   for v in row])
    ws.freeze_panes = "A2"

    # 12 个品评卡：Beer# 1–12
    for n in range(1, 13):
        ws = wb[f"beer_{n}"]
        if ws["S3"].value != n:
            raise ValueError(f"beer_{n} 的 Beer# 不是 {n}")
        row = table.loc[n]
        for s, (col, r) in SHEET_CELLS.items():
            ws[f"{col}{r}"] = int(row[s])
        # acetaldehyde 无感官标签：用化学浓度代理等级（非感官预测）
        ws["Q45"] = int(proxy.loc[n, "proxy_level"])
        ws["AG45"] = int(proxy.loc[n, "proxy_level"])
        # 未映射格清零，避免模板公式回显 0
        ws["AG62"] = None
        ws["AG94"] = None

    # summary：Beer 1–12 汇总
    ws = wb["summary"]
    hdr = [c.value for c in ws[1]]
    for n in range(1, 13):
        for j, name in enumerate(hdr[1:], start=2):
            key = SUMMARY_ALIAS.get(name, name)
            if key in table.columns:
                ws.cell(row=1 + n, column=j).value = int(table.loc[n, key])

    note = wb.create_sheet("说明")
    fm = pd.read_csv(C.OUT_DIR / "fallback_metrics.csv").set_index("target")
    lines = [
        ["字段", "说明"],
        ["来源", "46 维：results/relative_scores_v3.csv；4 维：results/fallback_levels_4.csv"],
        ["50 维口径", "全部为模型相对 0–6 等级：观测分布 1/7…6/7 分位分档；0=最低段，6=最高段"],
        ["主模型", "ET 多输出共享树结构，46 维 = 30 核心 + 16 辅助"],
        ["回退 4 维", "DMS/Diacetyl/4-VG/t2n：同一特征与 ET 配置的单维模型（未进入主模型）"],
        ["DMS 回退", f"R²={fm.loc['Dimethyl sulfide','r2_oof']:.3f}, "
                     f"Spearman={fm.loc['Dimethyl sulfide','spearman_oof']:.3f}（弱）"],
        ["Diacetyl 回退", f"R²={fm.loc['Diacetyl','r2_oof']:.3f}, "
                          f"Spearman={fm.loc['Diacetyl','spearman_oof']:.3f}（弱）"],
        ["4-VG 回退", f"R²={fm.loc['4-vinyl guaiacol','r2_oof']:.3f}, "
                      f"Spearman={fm.loc['4-vinyl guaiacol','spearman_oof']:.3f}（中等）"],
        ["t2n 回退", f"R²={fm.loc['Trans-2-nonenal','r2_oof']:.3f}, "
                     f"Spearman={fm.loc['Trans-2-nonenal','spearman_oof']:.3f}（弱）"],
        ["OAV 标志", "见 fault_flags 工作表；DMS/Diacetyl/4-VG 的超阈判读，不进入 0–6"],
        ["t2n 阈值", "8.xlsx 无阈值，因此只有模型等级，没有绝对超阈判读"],
        ["注意 1", "相对等级不是原始人工 0–6 分；禁止报告伪分级准确率/Kappa"],
        ["注意 2", "主模型 14/46 为弱维；4 个回退维中 DMS/Diacetyl/t2n 也为弱维"],
        ["主键", "Beer = 4.csv 的 beer 列（1–250）"],
        ["12 个 beer sheet", "按 Beer# 1–12 填入；acetaldehyde 不在 50 维标签内，保持空白"],
        ["acetaldehyde", "4.csv 无感官标签；用 1.csv 浓度分位生成化学代理 0–6（非感官预测）；OAV 全部 below（阈值 50 mg/L）"],
        ["acetaldehyde 位置", "Q45/AG45；Aroma/Flavour 使用同一代理等级（无标签无法区分）"],
        ["AG62/AG94", "AG62 为无标签的重复 aftertaste 格，AG94 为 Comments，均留空"],
        ["summary", "Beer 1–12 的 50 维汇总"],
    ]
    for row in lines:
        note.append(row)
    note.column_dimensions["A"].width = 24
    note.column_dimensions["B"].width = 100

    dst = C.ROOT / "附件" / "3_品评表_相对等级.xlsx"
    try:
        wb.save(dst)
    except PermissionError:
        dst = C.ROOT / "附件" / "3_品评表_相对等级_填写版.xlsx"
        wb.save(dst)
        print("主文件被占用（Excel 打开），已另存为副本")
    print(f"saved {dst}")
    print(f"rows={len(out)}, dims={len(orig_cols)}, "
          f"nonnull={int(table.notna().sum().sum())}/{table.size}")


if __name__ == "__main__":
    main()
