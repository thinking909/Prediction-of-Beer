"""Q3 Step 3：缺陷约束与诊断（生成版）。

生成阶段采用“风格 95% 分位”作为缺陷上界（step2 已做），保留风格自身的
缺陷分布；同时输出一个“阈值硬截断”的严格版本，供消融比较。

  - 主约束 constrained_profiles.csv：style 95% 分位封顶；
  - 严格版 strict_threshold_profiles.csv：OAV 封顶到 1（仅作消融）；
  - t2n 无阈值，只报告浓度；
  - Stale hops 无化学列，标记软约束。

输出：
  results/constrained_profiles.csv
  results/constrained_concentration.csv
  results/strict_threshold_profiles.csv
  results/strict_threshold_concentration.csv
  results/defect_summary.csv
"""
import numpy as np
import pandas as pd

import q3_config as C
import dataio


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    meta = pd.read_csv(C.OUT_DIR / "compound_meta.csv", encoding="utf-8-sig")
    is_log = dict(zip(meta.compound, meta.is_log.astype(bool)))

    gen = pd.read_csv(C.OUT_DIR / "generated_profiles.csv", encoding="utf-8-sig")
    genc = pd.read_csv(C.OUT_DIR / "generated_concentration.csv", encoding="utf-8-sig")

    oav_defects = {
        "Dimethyl sulfide": C.DEFECTS["Dimethyl sulfide"]["threshold"],
        "Diacetyl": C.DEFECTS["Diacetyl"]["threshold"],
        "4-vinylguaiacol": C.DEFECTS["4-vinylguaiacol"]["threshold"],
    }

    # 主约束：style 95% 已在 step2 完成，这里只做记录
    gen.to_csv(C.OUT_DIR / "constrained_profiles.csv", index=False, encoding="utf-8-sig")
    genc.to_csv(C.OUT_DIR / "constrained_concentration.csv", index=False, encoding="utf-8-sig")

    # 严格版：OAV 封顶到 1
    strict = gen.copy()
    for c, th in oav_defects.items():
        if c in cols:
            cap = np.log10(th) if is_log.get(c, False) else th
            strict[c] = np.minimum(strict[c].to_numpy(), cap)
    strict_c = strict.copy()
    for c in cols:
        if is_log.get(c, False):
            strict_c[c] = np.power(10.0, strict_c[c])
    strict.to_csv(C.OUT_DIR / "strict_threshold_profiles.csv",
                  index=False, encoding="utf-8-sig")
    strict_c.to_csv(C.OUT_DIR / "strict_threshold_concentration.csv",
                    index=False, encoding="utf-8-sig")

    rows = []
    for s in sorted(gen["style"].unique()):
        gc = genc[genc["style"] == s]
        sc = strict_c[strict_c["style"] == s]
        row = {"style": s, "n": len(gc)}
        for c, th in oav_defects.items():
            before = gc[c].to_numpy() / th
            after = sc[c].to_numpy() / th
            row[f"{c}_oav_med_style95"] = float(np.median(before))
            row[f"{c}_oav_p95_style95"] = float(np.quantile(before, 0.95))
            row[f"{c}_frac_gt1_style95"] = float(np.mean(before > 1))
            row[f"{c}_frac_gt1_strict"] = float(np.mean(after > 1))
        row["t2n_med"] = float(gc["E-2-nonenal"].median())
        row["t2n_p95"] = float(gc["E-2-nonenal"].quantile(0.95))
        row["stale_hops"] = "soft_constraint_no_chemical"
        rows.append(row)
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "defect_summary.csv",
                              index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows)[[
        "style",
        "Dimethyl sulfide_frac_gt1_style95", "Dimethyl sulfide_frac_gt1_strict",
        "Diacetyl_frac_gt1_style95", "Diacetyl_frac_gt1_strict",
        "4-vinylguaiacol_frac_gt1_style95", "4-vinylguaiacol_frac_gt1_strict",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
