"""Q2 Step 0：数据与口径校验。

只校验，不重跑清洗、不训练模型。
输出：Q2/results/q2_validation.json
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import q2_config as C
import dataio


def corr_pair(df, a, b, method="spearman"):
    x, y = df[a], df[b]
    mask = x.notna() & y.notna()
    if mask.sum() < 10:
        return np.nan
    f = spearmanr if method == "spearman" else pearsonr
    return float(f(x[mask], y[mask]).statistic)


def matrix_type_check(mat, source, pairs, seed=0):
    """对比矩阵值与重算值，判断矩阵是 Pearson 还是 Spearman。"""
    rng = np.random.default_rng(seed)
    rows = []
    for a, b in pairs:
        if a not in mat.index or b not in mat.columns:
            continue
        rows.append({
            "a": a, "b": b, "matrix": float(mat.loc[a, b]),
            "spearman": corr_pair(source, a, b, "spearman"),
            "pearson": corr_pair(source, a, b, "pearson"),
        })
    d = pd.DataFrame(rows).dropna()
    if d.empty:
        return {"n_pairs": 0}
    d["d_spearman"] = (d["matrix"] - d["spearman"]).abs()
    d["d_pearson"] = (d["matrix"] - d["pearson"]).abs()
    return {
        "n_pairs": int(len(d)),
        "median_abs_diff_spearman": float(d["d_spearman"].median()),
        "median_abs_diff_pearson": float(d["d_pearson"].median()),
        "frac_within_0.01_spearman": float((d["d_spearman"] < 0.01).mean()),
        "frac_within_0.01_pearson": float((d["d_pearson"] < 0.01).mean()),
        "examples": d.head(5)[["a", "b", "matrix", "spearman", "pearson"]]
                    .round(4).to_dict("records"),
    }


def matrix_type_check_two(mat, fa, fb, pairs, seed=0):
    """两个不同 DataFrame 的配对（如 6.csv：化学 × 感官）。"""
    rows = []
    for a, b in pairs:
        if a not in mat.index or b not in mat.columns:
            continue
        x, y = fa[a], fb[b]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        rows.append({
            "a": a, "b": b, "matrix": float(mat.loc[a, b]),
            "spearman": float(spearmanr(x[mask], y[mask]).statistic),
            "pearson": float(pearsonr(x[mask], y[mask]).statistic),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n_pairs": 0}
    d["d_spearman"] = (d["matrix"] - d["spearman"]).abs()
    d["d_pearson"] = (d["matrix"] - d["pearson"]).abs()
    return {
        "n_pairs": int(len(d)),
        "median_abs_diff_spearman": float(d["d_spearman"].median()),
        "median_abs_diff_pearson": float(d["d_pearson"].median()),
        "frac_within_0.01_spearman": float((d["d_spearman"] < 0.01).mean()),
        "frac_within_0.01_pearson": float((d["d_pearson"] < 0.01).mean()),
        "examples": d.head(5)[["a", "b", "matrix", "spearman", "pearson"]]
                    .round(4).to_dict("records"),
    }


def missing_pair_check(mat, fa, fb, fa_clean, fb_clean, miss_cols, other_cols, seed=0, n=30):
    rng = np.random.default_rng(seed)
    pairs = []
    for m in miss_cols:
        if m not in mat.index:
            continue
        for _ in range(3):
            o = other_cols[int(rng.integers(0, len(other_cols)))]
            if o in mat.columns and o != m:
                pairs.append((m, o))
    pairs = pairs[:n]
    rows = []
    for a, b in pairs:
        x, y = fa[a], fb[b]
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        xc, yc = fa_clean[a], fb_clean[b]
        rows.append({
            "matrix": float(mat.loc[a, b]),
            "raw": float(spearmanr(x[mask], y[mask]).statistic),
            "cleaned": float(spearmanr(xc, yc).statistic),
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return {"n_pairs": 0}
    d["d_raw"] = (d["matrix"] - d["raw"]).abs()
    d["d_cleaned"] = (d["matrix"] - d["cleaned"]).abs()
    return {
        "n_pairs": int(len(d)),
        "median_abs_diff_raw": float(d["d_raw"].median()),
        "median_abs_diff_cleaned": float(d["d_cleaned"].median()),
        "frac_within_0.01_raw": float((d["d_raw"] < 0.01).mean()),
        "frac_within_0.01_cleaned": float((d["d_cleaned"] < 0.01).mean()),
    }


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    chem = b.chem
    sens = b.sens
    scale = b.scale

    m2 = pd.read_csv(C.DATA_DIR / "2_化学参数Spearman相关矩阵.csv", index_col=0, encoding="utf-8-sig")
    m6 = pd.read_csv(C.DATA_DIR / "6_化学感官参数Spearman相关矩阵.csv", index_col=0, encoding="utf-8-sig")
    clean1 = pd.read_csv(C.DATA_DIR / "1_化学测量数据.csv", index_col="beer",
                         encoding="utf-8-sig").sort_index()
    # 清洗表保留原始缺失；本诊断需完整矩阵做 Spearman，故临时按列中位数填补
    clean1 = clean1.fillna(clean1.median(numeric_only=True))

    chem_names = list(chem.columns)
    sens_names = [c for c in sens.columns if c != C.STRATIFY_COL]

    # ---------- 名称对齐 ----------
    names = {
        "chem_cols": len(chem_names),
        "sens_cols": len(sens_names),
        "m2_index_not_in_chem": sorted(set(m2.index) - set(chem_names)),
        "m2_cols_not_in_chem": sorted(set(m2.columns) - set(chem_names)),
        "chem_not_in_m2": sorted(set(chem_names) - set(m2.index)),
        "m6_rows_not_in_chem": sorted(set(m6.index) - set(chem_names)),
        "m6_cols_not_in_sens": sorted(set(m6.columns) - set(sens_names)),
        "sens_not_in_m6": sorted(set(sens_names) - set(m6.columns)),
    }

    # ---------- 2.csv 类型与来源 ----------
    rng = np.random.default_rng(0)
    names2 = [n for n in m2.index if n in chem.columns]
    pairs2 = [(names2[i], names2[j]) for i, j in
              rng.integers(0, len(names2), size=(40, 2)) if i != j][:30]
    type2_raw = matrix_type_check(m2, chem, pairs2, seed=0)
    type2_clean = matrix_type_check(m2, clean1, pairs2, seed=0)

    # ---------- 6.csv 类型 ----------
    rows6 = [n for n in m6.index if n in chem.columns]
    cols6 = [n for n in m6.columns if n in sens.columns]
    pairs6 = [(rows6[i], cols6[j]) for i, j in
              rng.integers(0, len(rows6), size=(40, 2))][:20]
    pairs6 = [(a, b) for a, b in pairs6 if a in chem.columns and b in sens.columns]
    type6_raw = matrix_type_check_two(m6, chem, sens, pairs6, seed=1)

    miss_cols = [c for c in chem.columns if chem[c].isna().any()]
    other_chem = [c for c in chem.columns if c not in miss_cols]
    miss2 = missing_pair_check(m2, chem, chem, clean1, clean1,
                               miss_cols, other_chem, seed=2)
    miss6 = missing_pair_check(m6, chem, sens, clean1, sens,
                               [c for c in miss_cols if c in m6.index],
                               sens_names, seed=3)

    # ---------- 缺失 / 常量 / 重复 ----------
    miss = chem.isna().sum()
    views = {}
    for v in ("V2_scale_aware", "V3_absolute"):
        X = b.X(v)
        views[v] = {
            "n_cols": int(X.shape[1]),
            "n_nan": int(X.isna().sum().sum()),
            "n_constant": int((X.std(ddof=0) == 0).sum()),
        }
    arr = b.X("V2_scale_aware").to_numpy()
    seen, dups = {}, []
    for j, c in enumerate(b.X("V2_scale_aware").columns):
        key = arr[:, j].tobytes()
        if key in seen:
            dups.append([seen[key], c])
        else:
            seen[key] = c

    # ---------- 目标 ----------
    targets = {
        "core": len(C.CORE_TARGETS),
        "aux": len(C.AUX_TARGETS),
        "model_46": len(C.MODEL_TARGETS),
        "defect_4": len(C.DEFECT_TARGETS),
        "all_50": len(sens_names),
        "missing_core": [t for t in C.CORE_TARGETS if t not in sens.columns],
        "missing_aux": [t for t in C.AUX_TARGETS if t not in sens.columns],
        "missing_defect": [t for t in C.DEFECT_TARGETS if t not in sens.columns],
        "acetaldehyde_in_sens": any("acetald" in c.lower() for c in sens.columns),
    }
    label_stats = {
        "mean_abs_max": float(sens[sens_names].mean().abs().max()),
        "sd_min": float(sens[sens_names].std().min()),
        "sd_max": float(sens[sens_names].std().max()),
        "min": float(sens[sens_names].min().min()),
        "max": float(sens[sens_names].max().max()),
    }

    # ---------- 风格 ----------
    style_counts = sens[C.STRATIFY_COL].value_counts()
    styles = {
        "n_styles": int(style_counts.size),
        "min_n": int(style_counts.min()),
        "n_ge_8": int((style_counts >= C.STYLE_MIN_N).sum()),
        "counts": {k: int(v) for k, v in style_counts.items()},
    }

    report = {
        "files": {
            "chem_shape": list(chem.shape),
            "sens_shape": list(sens.shape),
            "m2_shape": list(m2.shape),
            "m6_shape": list(m6.shape),
            "derived_cols": C.DERIVED_COLS,
        },
        "names": names,
        "correlation_type": {
            "2csv_vs_raw": type2_raw,
            "2csv_vs_cleaned": type2_clean,
            "6csv_vs_raw": type6_raw,
            "2csv_missing_pairs": miss2,
            "6csv_missing_pairs": miss6,
        },
        "m2": {
            "symmetric": bool(np.allclose(m2.to_numpy(), m2.to_numpy().T, equal_nan=True)),
            "diag_all_one": bool(np.allclose(np.diag(m2.to_numpy()), 1.0)),
            "range": [float(np.nanmin(m2.to_numpy())), float(np.nanmax(m2.to_numpy()))],
        },
        "missing": {
            "total": int(miss.sum()),
            "n_cols": int((miss > 0).sum()),
            "columns": {k: int(v) for k, v in miss[miss > 0].items()},
        },
        "views": views,
        "duplicate_columns": dups,
        "targets": targets,
        "label_stats": label_stats,
        "styles": styles,
    }
    (C.OUT_DIR / "q2_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "2csv_median_diff_spearman_raw": type2_raw.get("median_abs_diff_spearman"),
        "2csv_median_diff_pearson_raw": type2_raw.get("median_abs_diff_pearson"),
        "2csv_median_diff_spearman_cleaned": type2_clean.get("median_abs_diff_spearman"),
        "6csv_median_diff_spearman_raw": type6_raw.get("median_abs_diff_spearman"),
        "missing": int(miss.sum()),
        "duplicates": len(dups),
        "name_issues": sum(len(v) for k, v in names.items() if isinstance(v, list)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
