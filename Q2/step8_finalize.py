"""Q2 Step 8：最终 Top-3 + 化学合理性。

输入 stage3，补：
- 来源/单位/阈值/OAV（绝对列）
- 已知风味/通路（人工映射，未知标 none）
- 最终状态：stable / exploratory / definitional / not_supported
输出：Q2/results/final/*
"""
import json

import numpy as np
import pandas as pd

import q2_config as C
import dataio

FLAVOR = {
    "Lactic acid": "yogurt, sour, creamy",
    "Acetic acid": "vinegar, sour",
    "pH": "acidity/sourness",
    "Iso-alpha acids": "bitter, hop",
    "Ethanol": "alcohol, warming",
    "Glycerol": "body, sweet",
    "Protein": "body, mouthfeel",
    "Color": "visual (not aroma)",
    "Caloric value": "body/energy (not aroma)",
    "Carbon dioxide": "carbonation, prickly",
    "Sugars": "sweet, body",
    "Iron": "metallic",
    "Unfermentables": "body, residual sweetness",
    "4-vinylguaiacol": "clove, smoky, medicinal",
    "Diacetyl": "butter, butterscotch",
    "Dimethyl sulfide": "cooked corn, cabbage",
    "E-2-nonenal": "wet cardboard, papery",
    "Metallic": "metallic",
    "Stale hops": "cheesy, old hops",
    "Linalool": "floral, lavender, citrus",
    "Geraniol": "rose, floral",
    "Citronellol": "citrus, rose",
    "Alpha-terpineol": "floral, lilac",
    "Nerol": "rose, citrus",
    "Beta-phellandrene": "citrus, mint, pepper",
    "Alpha-phellandrene": "citrus, herbal",
    "Limonene": "citrus, orange",
    "Caryophyllene": "spicy, woody, clove",
    "Alpha-humulene": "woody, hop",
    "Beta-humulene": "woody, hop",
    "E-Beta-farnesene": "green, hop",
    "Z-Beta-farnesene": "green, hop",
    "Germacrene D": "green, woody",
    "Valencene": "citrus, orange",
    "E-Beta-ionone": "violet, floral",
    "E-Alpha-ionone": "violet, floral",
    "E-Beta-damascenone": "apple, rose, cooked",
    "Geosmin": "earthy, beetroot",
    "Eucalyptol": "mint, eucalyptus",
    "Camphor": "camphor, medicinal",
    "Terpinen-4-ol": "tea tree, spicy",
    "Alpha-calacorene": "woody, hop",
    "Caryophylla-4,8-dien-5-ol": "woody, spicy",
    "Ethyl acetate": "solvent, fruity",
    "Isoamyl acetate": "banana, pear",
    "Isobutyl acetate": "fruity, banana",
    "Phenethyl acetate": "rose, honey",
    "Ethyl butyrate": "fruity, pineapple",
    "Ethyl hexanoate": "apple, fruity",
    "Ethyl octanoate": "fruity, wine, apple",
    "Ethyl decanoate": "fruity, grape",
    "Ethyl 3-methylbutyrate": "fruity, apple",
    "Ethyl 2-methylbutyrate": "fruity, apple",
    "Ethyl isobutyrate": "fruity",
    "Ethyl lactate": "fruity, creamy",
    "Ethyl succinate": "fruity, wine",
    "Diethyl succinate": "fruity, wine",
    "Ethyl benzoate": "fruity, balsamic",
    "Ethyl cinnamate": "cinnamon, balsamic",
    "Ethyl phenylacetate": "honey, floral",
    "Octyl acetate": "fruity, floral",
    "Heptyl acetate": "fruity, floral",
    "Amyl acetate": "banana, pear",
    "Isopentyl hexanoate": "fruity, apple",
    "Pentyl octanoate": "fruity, waxy",
    "2-methylbutyl octanoate": "fruity",
    "Benzyl alcohol": "floral, rose",
    "Benzaldehyde": "almond, cherry",
    "Furfural": "toasted, caramel, cardboard",
    "5-methylfurfural": "caramel, toasted",
    "Furfuryl alcohol": "burnt, caramel",
    "2-acetylfuran": "toasted, coffee",
    "Methylpyrazine": "roasted, nutty",
    "2,6-dimethylpyrazine": "roasted, nutty",
    "2-acetylpyrrole": "toasted, nutty",
    "Isomaltol": "caramel, sweet",
    "Methanethiol": "cooked cabbage, sulfur",
    "Hydrogen sulfide": "rotten egg, sulfur",
    "Carbon disulfide": "solvent, sulfur",
    "Dimethyl disulfide": "garlic, sulfur",
    "Diethyl sulfide": "onion, sulfur",
    "Sulfur dioxide": "sulfite, pungent",
    "Isoamyl alcohol": "fusel, alcoholic",
    "Isobutanol": "fusel, alcoholic",
    "2-methyl-1-butanol": "fusel, alcoholic",
    "N-propanol": "alcohol",
    "N-pentanol": "alcohol, fruity",
    "Phenylacetaldehyde": "honey, rose",
    "E,E-2,4-decadienal": "fried, fatty",
    "O-cresol": "phenolic, medicinal",
    "Eugenol": "clove, spicy",
    "Vanillin": "vanilla, sweet",
    "4-ethylguaiacol": "smoky, spicy",
    "4-ethyl-phenol": "phenolic, barnyard",
}

PATHWAY = {
    "ester": "yeast esterification",
    "linalool": "hop-derived terpene",
    "geraniol": "hop-derived terpene",
    "citronellol": "hop-derived terpene",
    "terpineol": "hop-derived terpene",
    "nerol": "hop-derived terpene",
    "phellandrene": "hop-derived terpene",
    "limonene": "hop-derived terpene",
    "caryophyllene": "hop-derived terpene",
    "humulene": "hop-derived terpene",
    "farnesene": "hop-derived terpene",
    "ionone": "hop/carotenoid-derived terpenoid",
    "damascenone": "hop/carotenoid-derived terpenoid",
    "geosmin": "microbial terpene",
    "cresol": "phenolic off-flavor",
    "guaiacol": "phenolic off-flavor",
    "vinylguaiacol": "phenolic off-flavor",
    "ethylphenol": "phenolic off-flavor",
    "diacetyl": "yeast acetoin/butanediol metabolism",
    "dimethyl sulfide": "malt sulfur metabolism",
    "nonenal": "lipid oxidation",
    "methanethiol": "yeast sulfur metabolism",
    "hydrogen sulfide": "yeast sulfur metabolism",
    "disulfide": "yeast sulfur metabolism",
    "sulfide": "yeast sulfur metabolism",
    "furfural": "malt Maillard reaction",
    "pyrazine": "malt Maillard reaction",
    "pyrrole": "malt Maillard reaction",
    "lactic acid": "lactic fermentation",
    "acetic acid": "acetic acid bacteria / wild flora",
    "ethanol": "yeast fermentation",
    "glycerol": "yeast fermentation",
    "sugars": "malt",
    "protein": "malt",
    "iron": "malt/process",
    "color": "malt/process",
}


def lookup_flavor(c):
    if c in FLAVOR:
        return FLAVOR[c], "specific"
    low = c.lower()
    for key in PATHWAY:
        if key in low:
            return f"class-level: {key}", "class"
    return "no documented descriptor", "none"


def lookup_pathway(c):
    low = c.lower()
    for key, val in PATHWAY.items():
        if key in low:
            return val
    return ""


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "final"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    stage3 = pd.read_csv(C.OUT_DIR / "style" / "q2_top3_stage3.csv")
    scale = pd.read_csv(C.DATA_DIR / "化学字段单位与尺度表.csv", encoding="utf-8-sig")
    scale = scale.rename(columns={"来源(origin_paper)": "origin"})
    smap = scale.set_index("列名")

    rows = []
    for _, r in stage3.iterrows():
        c = r.compound
        meta = smap.loc[c] if c in smap.index else None
        absolute = meta is not None and meta["绝对/相对"] == "绝对"
        thr = float(meta["阈值均值"]) if meta is not None and pd.notna(meta["阈值均值"]) else np.nan
        unit = meta["单位"] if meta is not None else ""
        origin = meta["origin"] if meta is not None else ""
        oav_med = np.nan
        oav_gt1 = np.nan
        if absolute and np.isfinite(thr) and c in b.chem_restored.columns:
            oav = (b.chem_restored[c] / thr).dropna()
            oav_med = float(oav.median())
            oav_gt1 = float((oav > 1).mean())
        flavor, level = lookup_flavor(c)
        pathway = lookup_pathway(c)
        has_chem = (level != "none") or np.isfinite(thr) or bool(pathway)
        status = r.status_stage3
        if status == "stable_candidate":
            status = "stable" if has_chem else "exploratory"
        elif status == "exploratory":
            if (pd.notna(r.univariate_q) and r.univariate_q > 0.05 and
                    not bool(r.drop_pass) and
                    (pd.isna(r.sign_consistency) or r.sign_consistency < 0.8)):
                status = "not_supported"
        reliability = ("high" if status == "stable" else
                       "medium" if status == "definitional" or
                       (pd.notna(r.univariate_q) and r.univariate_q < 0.05) else "low")
        rows.append({
            **r.to_dict(),
            "origin": origin,
            "unit": unit,
            "absolute_or_relative": "absolute" if absolute else "relative",
            "threshold": thr,
            "oav_median": oav_med,
            "oav_gt1_ratio": oav_gt1,
            "known_flavor": flavor,
            "descriptor_level": level,
            "pathway": pathway,
            "has_chem_evidence": bool(has_chem),
            "status_final": status,
            "reliability": reliability,
        })
    res = pd.DataFrame(rows)
    res["sign_adjusted"] = np.sign(res.rho_adjusted)
    res["sign_conflict"] = ((res.sign != res.sign_adjusted) &
                            (res.rho_adjusted.abs() > 0))
    res["borderline"] = res.rho_adjusted.abs() < 0.15
    # 风格调整后方向翻转：视为方向不稳，降为 exploratory
    mask = (res.status_final == "stable") & res.sign_conflict
    res.loc[mask, "status_final"] = "exploratory"
    res.loc[mask, "reliability"] = "low"
    res.loc[(res.status_final == "stable") & res.borderline, "reliability"] = "medium"
    res.to_csv(out / "q2_top3_final.csv", index=False)

    # 目标级稳定性汇总
    summary_rows = []
    for (v, t), d in res.groupby(["view", "target"]):
        stable = d[d.status_final == "stable"]
        summary_rows.append({
            "view": v, "target": t,
            "n_stable": int((d.status_final == "stable").sum()),
            "stable_compounds": " | ".join(stable.sort_values("rank").compound),
            "definitional": int((d.status_final == "definitional").sum()),
            "exploratory": int((d.status_final == "exploratory").sum()),
            "not_supported": int((d.status_final == "not_supported").sum()),
            "style_proxy": int((d.style_flag == "style_proxy").sum()),
            "drop_pass": bool(d.drop_pass.iloc[0]),
            "max_adjusted_rho": float(d.rho_adjusted.abs().max()),
        })
    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "q2_stability_final.csv", index=False)

    summary = {}
    for v in ["V3", "V2"]:
        d = res[res.view == v]
        summary[v] = {
            "status_final": d.status_final.value_counts().to_dict(),
            "targets_with_stable": int(d[d.status_final == "stable"].target.nunique()),
            "descriptor_level": d.descriptor_level.value_counts().to_dict(),
            "has_chem_evidence": float(d.has_chem_evidence.mean()),
            "n_stable_rows": int((d.status_final == "stable").sum()),
        }
    (out / "step8_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
