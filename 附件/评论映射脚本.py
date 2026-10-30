# -*- coding: utf-8 -*-
"""
评论指标(7.csv) -> 专业感官指标(4.csv) 映射表生成
依据:
  A 官方 vetted 映射: 8.xlsx 'Sensory names' 双数据集按
    (Category_Paper, CleanName_Paper, CleanName_PaperLong, CleanName_PanelFeedback) 配对
    （等价于论文 Main Figure Generator.Rmd 的 map_panel_ratebeer pivot_wider）
  B 论文 Fig3 实际配对 (ep_rb_plot 调用, 19 对)
  C 语义/化学推断: 依据 3.xlsx 品评卡描述符与啤酒风味化学，逐条标注置信度
输出: 清洗后/评论指标映射.csv (+ 控制台统计)
"""
import pandas as pd
from pathlib import Path

BASE = str(Path(__file__).resolve().parent / "清洗后")
d7 = pd.read_csv(BASE + "/7_论坛评论数据.csv", encoding="utf-8-sig")
d4 = pd.read_csv(BASE + "/4_感官分析结果.csv", encoding="utf-8-sig")
sn = pd.read_excel(BASE + "/8_其他相关数据.xlsx", sheet_name="Sensory names")
PANEL = [c for c in d4.columns if c not in ("beer", "beer_id", "tasting_category_fine")]

def norm(s):
    s = str(s).replace("_x000D_", " ").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(s.split())

ep = sn[sn["dataset"] == "Trained panel"].copy()
rb = sn[sn["dataset"] == "Ratebeer"].copy()
ep["clean_col"] = ep["CleanName_PanelFeedback"].map(norm)
panel_map = {r["R_name"]: r["clean_col"] for _, r in ep.iterrows()
             if pd.notna(r["R_name"]) and r["clean_col"] in PANEL}
assert len(panel_map) == 50, f"panel_map={len(panel_map)}"

key = ["Category_Paper", "CleanName_Paper", "CleanName_PaperLong", "CleanName_PanelFeedback"]
official = {}
for k, grp in rb.groupby(key, dropna=False):
    hits = ep
    for col, val in zip(key, k):
        hits = hits[hits[col].isna()] if pd.isna(val) else hits[hits[col] == val]
    targets = [panel_map[t] for t in hits["R_name"].dropna() if t in panel_map]
    if targets:
        for rname in grp["R_name"].dropna():
            official[rname] = {
                "targets": targets,
                "clean": "" if pd.isna(k[1]) else str(k[1]),
                "long": "" if pd.isna(k[2]) else str(k[2]),
            }

FIG3 = {("Alcohol", "taste_alcohol"), ("Acidity", "taste_sour"), ("Bitterness", "taste_bitter"),
        ("Sweetness", "taste_sweet"), ("Malt aroma - overall", "aroma_malt"),
        ("Malt taste - overall", "taste_malt"), ("Hops aroma - overall", "aroma_hoppy"),
        ("Hops taste - overall", "taste_hoppy"), ("Esters aroma - overall", "aroma_estery"),
        ("Esters taste - overall", "taste_estery"), ("4-vinyl guaiacol", "aroma_clove"),
        ("Malt taste - caramel", "taste_caramel"), ("Carbonation", "palate_carbonation"),
        ("Aftertaste", "palate_aftertaste"), ("Coriander", "aroma_coriander"),
        ("Diacetyl", "aroma_buttery"), ("Malt aroma - smoked", "aroma_burnt"),
        ("Hops aroma - tropical fruit", "aroma_tropical"), ("Esters aroma - banana", "aroma_banana")}

SINGLE = {"4-vinyl guaiacol", "Clove", "Acetic acid", "Lactic acid", "Dimethyl sulfide",
          "Diacetyl", "Metallic", "Barnyard", "Coriander", "Orange", "Stale hops",
          "Trans-2-nonenal", "Sweetness", "Bitterness", "Acidity", "Alcohol", "Aftertaste",
          "Carbonation", "Overall appreciation (Trained panel)", "Body"}

INFERRED = {
    "acetaldehyde": ([], "4.csv 无乙醛面板列; 对应化学指标 Acetaldehyde", "无对应"),
    "apple": ([], "官方 taste_apple→Acetaldehyde; 4.csv 无该面板列, 仅化学指标 Acetaldehyde", "无对应"),
    "acetate": (["Esters aroma - nail polish", "Esters taste - nail polish"], "乙酸乙酯/溶剂味", "推断-高"),
    "acidic": (["Acidity"], "酸感", "推断-高"),
    "acetic": (["Acetic acid"], "乙酸", "推断-高"),
    "anise": (["Clove"], "品评卡 'Clove/anise' 合并项", "推断-高"),
    "blanket": (["Trans-2-nonenal"], "湿毯味属老化味(纸板类)", "推断-中"),
    "burnt": (["Malt aroma - smoked", "Malt taste - smoked"], "品评卡 'Smoked, burnt toast'", "推断-高"),
    "butterscotch": (["Diacetyl"], "奶油糖味=双乙酰", "推断-高"),
    "cabbage": (["Dimethyl sulfide"], "煮白菜味=DMS", "推断-高"),
    "candy": (["Esters aroma - banana", "Esters taste - banana"], "品评卡 'Banana, pear, candy'", "推断-高"),
    "cardboard": (["Trans-2-nonenal"], "湿纸板味", "推断-高"),
    "cereal": (["Malt aroma - grain", "Malt taste - grain"], "品评卡 'Cereals, flour, grains'", "推断-高"),
    "cheesy": (["Stale hops"], "奶酪/蓝纹味", "推断-高"),
    "dentist": (["4-vinyl guaiacol"], "品评卡 4VG 描述含 dentist", "推断-高"),
    "diacetyl": (["Diacetyl"], "直接对应", "推断-高"),
    "flour": (["Malt aroma - grain", "Malt taste - grain"], "品评卡 'Cereals, flour, grains'", "推断-高"),
    "grape": (["Esters aroma - fruity", "Esters taste - fruity"], "葡萄香归果香酯类, 置信度低", "推断-低"),
    "hay": (["Hops aroma - grassy", "Hops taste - grassy"], "品评卡 noble hops(干草)", "推断-高"),
    "herb": (["Hops aroma - grassy", "Hops taste - grassy"], "品评卡 noble hops(干草药)", "推断-高"),
    "horse": (["Barnyard"], "马厩味", "推断-高"),
    "kiwi": (["Hops aroma - tropical fruit", "Hops taste - tropical fruit"], "热带果", "推断-中"),
    "lactic": (["Lactic acid"], "乳酸", "推断-高"),
    "lychee": (["Hops aroma - tropical fruit", "Hops taste - tropical fruit"], "热带果", "推断-中"),
    "mango": (["Hops aroma - tropical fruit", "Hops taste - tropical fruit"], "热带果", "推断-中"),
    "medicinal": (["4-vinyl guaiacol"], "品评卡 4VG 描述含 medicinal", "推断-高"),
    "orange": (["Orange"], "橙/柑橘", "推断-高"),
    "papaya": (["Hops aroma - tropical fruit", "Hops taste - tropical fruit"], "热带果", "推断-中"),
    "papery": (["Trans-2-nonenal"], "纸味", "推断-高"),
    "pine": (["Hops aroma - woody", "Hops taste - woody"], "品评卡 'Pine, woody, resinous'", "推断-高"),
    "pineapple": (["Hops aroma - tropical fruit", "Hops taste - tropical fruit"], "热带果", "推断-中"),
    "resinous": (["Hops aroma - woody", "Hops taste - woody"], "品评卡 'Pine, woody, resinous'", "推断-高"),
    "solvent": (["Esters aroma - nail polish", "Esters taste - nail polish"], "品评卡 'Ethyl acetate, solvent'", "推断-高"),
    "toast": (["Malt aroma - smoked", "Malt taste - smoked"], "品评卡 'burnt toast'", "推断-高"),
    "stale": (["Trans-2-nonenal"], "官方表标注 CleanName=Trans-2-nonenal; 因 aroma/taste 的 Long 名不一致未自动配对, 手工补全", "推断-高"),
}
SCORES = {
    "aroma_10": ([], "评论端 Aroma 评分(面板无对应)", "分数-不可映射"),
    "taste_10": ([], "评论端 Taste 评分(面板无对应)", "分数-不可映射"),
    "palate_5": ([], "评论端口感评分(面板无综合对应项)", "分数-不可映射"),
    "appearance_5": ([], "外观评分(面板未评外观)", "分数-不可映射"),
    "ave_score_5": ([], "RateBeer 综合分(面板无对应)", "分数-不可映射"),
    "overall_20": (["Overall appreciation (Trained panel)"], "官方: Overall appreciation", "官方"),
}


def pick(col, targets):
    if col.startswith("aroma_"):
        pref = "Aroma"
    else:
        pref = "Taste"
    out = []
    for t in targets:
        if t in SINGLE:
            out.append(t)
        elif t.startswith("Malt aroma - ") or t.startswith("Hops aroma - ") or t.startswith("Esters aroma - "):
            out.append(t.replace(" aroma - ", f" {pref.lower()} - ") if pref == "Taste" else t)
        elif t.startswith("Malt taste - ") or t.startswith("Hops taste - ") or t.startswith("Esters taste - "):
            out.append(t.replace(" taste - ", " aroma - ") if pref == "Aroma" else t)
        else:
            out.append(t)
    kept, seen = [], set()
    for t in out:
        if t not in seen:
            seen.add(t)
            kept.append(t)
    return kept


def swap_side(name):
    if " aroma - " in name:
        return name.replace(" aroma - ", " taste - ")
    if " taste - " in name:
        return name.replace(" taste - ", " aroma - ")
    return name


def side_of(targets):
    if any(" aroma - " in t for t in targets):
        return "aroma"
    if any(" taste - " in t for t in targets):
        return "taste"
    return "any"


official_by_desc = {}
for full_col, info in official.items():
    desc = full_col.split("_", 1)[1]
    slot = official_by_desc.setdefault(desc, {})
    slot.setdefault(side_of(info["targets"]), []).extend(info["targets"])


def official_lookup(col):
    if col in official:
        return list(official[col]["targets"]), "官方", f"CleanName={official[col]['clean']}"
    d = col.split("_", 1)[1]
    if d not in official_by_desc:
        return None, "", ""
    slot = official_by_desc[d]
    aspect = col.split("_", 1)[0]
    if "any" in slot:
        return list(slot["any"]), "官方(同描述符跨面)", f"官方单属性配对 CleanName={d}"
    if aspect == "aroma" and "aroma" in slot:
        return list(slot["aroma"]), "官方", "官方 aroma 侧配对"
    if aspect in ("taste", "palate") and "taste" in slot:
        return list(slot["taste"]), "官方", "官方 taste 侧配对" + ("; palate列按taste侧归并" if aspect == "palate" else "")
    side = "aroma" if aspect == "aroma" else "taste"
    other = "taste" if side == "aroma" else "aroma"
    if other in slot:
        conv = [swap_side(t) for t in slot[other]]
        return conv, "官方(跨面推导)", f"官方仅有{other}侧配对, 按同描述符推导{side}侧"
    return None, "", ""


rows = []
for col in d7.columns:
    if col in ("beer", "beer_id", "tasting_category_fine"):
        continue
    if col in SCORES:
        tg, note, tier = SCORES[col]
        final = list(tg)
    else:
        tg, tier, note = official_lookup(col)
        if tg is None:
            d = col.split("_", 1)[1]
            if d in INFERRED:
                tg2, note, tier = INFERRED[d]
                final = pick(col, tg2)
            else:
                final, note, tier = [], "待人工确认", "未映射"
        else:
            final = list(tg)
            if col.endswith("_fruity"):
                note += "; 注意: 论文将fruity归到 nail polish(溶剂), 属官方原样"
            if any((t, col) in FIG3 for t in final):
                note += "; Fig3 验证配对"
    rows.append({"评论列(7.csv)": col,
                 "方面": col.split("_", 1)[0] if "_" in col else "",
                 "描述符": col.split("_", 1)[1] if "_" in col else col,
                 "对应专业列(4.csv)": ";".join(final) if final else "(无)",
                 "映射层级": tier, "依据/备注": note})

mp = pd.DataFrame(rows)
mp.to_csv(BASE + "/评论指标映射.csv", index=False, encoding="utf-8-sig")

used = {t for v in mp["对应专业列(4.csv)"] if v != "(无)" for t in v.split(";")}
print("评论列总数:", len(mp), "| 层级分布:", mp["映射层级"].value_counts().to_dict())
print("覆盖专业列:", len(used), "/", len(PANEL))
print("无评论对应的专业列:", [c for c in PANEL if c not in used])

REV = {}
for _, r in mp.iterrows():
    if r["对应专业列(4.csv)"] == "(无)":
        continue
    for pc in r["对应专业列(4.csv)"].split(";"):
        REV.setdefault(pc, []).append(r["评论列(7.csv)"])

print()
print("官方映射明细:")
print(mp[mp["映射层级"].str.startswith("官方")][["评论列(7.csv)", "对应专业列(4.csv)", "依据/备注"]].to_string(index=False))
print()
print("未映射/无对应:")
print(mp[mp["映射层级"].isin(["无对应", "未映射", "分数-不可映射"])][["评论列(7.csv)", "映射层级"]].to_string(index=False))
