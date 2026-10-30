"""步骤1：数据校验。产出 results/validation_report.json。

只做读取与一致性检查，不训练模型。
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
import dataio


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    rep = {}

    miss = b.chem.isna().sum()
    rep["matrix"] = {
        "n_samples": int(len(b.beer)),
        "n_chem_cols": int(b.chem.shape[1]),
        "n_sens_cols": int(len(C.AUX_ALL) + len(C.CORE_TARGETS)),
        "n_comment_cols": int(b.comments.shape[1]),
        "id_aligned": bool(list(b.chem.index) == list(b.sens.index) == list(b.comments.index)),
        "missing_chem": int(miss.sum()),
        "missing_chem_columns": {k: int(v) for k, v in miss[miss > 0].items()},
        "missing_sens": int(b.sens[C.AUX_ALL + C.CORE_TARGETS].isna().sum().sum()),
    }

    # ---- 视图列数
    views = {}
    expect = {"V1_paper_all": 231, "V2_scale_aware": 227, "V3_absolute": 70,
              "V4_relative": 157, "V5_restored": 227}
    for v in C.VIEWS:
        cols = b.feature_cols(v)
        X = b.X(v)
        views[v] = {
            "n_cols": len(cols),
            "expected": expect[v],
            "n_nan": int(X.isna().sum().sum()),
            "n_inf": int(np.isinf(X.to_numpy(dtype=float)).sum()),
            "n_constant": int((X.std(ddof=0) == 0).sum()),
            "n_negative": int((X < 0).sum().sum()),
        }
    rep["views"] = views

    # ---- V5 是否只是 V2 的单调变换（线性/核模型对照组的意义）
    v2, v5 = b.X("V2_scale_aware"), b.X("V5_restored")
    rho_min = 1.0
    for c in v2.columns:
        r = spearmanr(v2[c], v5[c], nan_policy="omit").statistic
        if np.isfinite(r):
            rho_min = min(rho_min, r)
    rep["V5_monotone_vs_V2"] = {"min_spearman": float(rho_min)}

    # 与预处理产出的还原文件交叉核对（非派生列、观测单元）
    try:
        provided = pd.read_csv(C.DATA_DIR / "1_化学测量数据_还原.csv", encoding="utf-8-sig") \
            .set_index(C.ID_COL).sort_index()
        chk_cols = [c for c in b.feature_cols("V2_scale_aware")]
        mism = []
        for c in chk_cols:
            mask = b.chem_restored[c].notna()
            if not np.allclose(b.chem_restored.loc[mask, c], provided.loc[mask, c],
                               rtol=1e-9, atol=0):
                mism.append(c)
        rep["restored_crosscheck"] = {"checked_cols": len(chk_cols), "mismatch": mism[:10]}
    except Exception as e:  # noqa: BLE001
        rep["restored_crosscheck"] = {"error": str(e)}

    # ---- 完全重复列（信息冗余的直接证据）
    arr = b.X("V1_paper_all").to_numpy(dtype=float)
    dup_pairs = []
    seen = {}
    for j, c in enumerate(b.X("V1_paper_all").columns):
        key = arr[:, j].tobytes()
        if key in seen:
            dup_pairs.append([seen[key], c])
        else:
            seen[key] = c
    rep["duplicate_columns"] = dup_pairs

    # ---- 标签
    Y = b.Y("model")
    lab = {
        "n_targets": int(Y.shape[1]),
        "mean_abs_max": float(Y.mean().abs().max()),
        "std_range": [float(Y.std().min()), float(Y.std().max())],
        "min": float(Y.min().min()), "max": float(Y.max().max()),
    }
    rep["labels"] = lab

    # ---- 尺度表
    s = b.scale
    rep["scale"] = {
        "by_category": s["尺度类别"].value_counts().to_dict(),
        "by_abs_rel": s["绝对/相对"].value_counts().to_dict(),
        "threshold_available": int((s["可做阈值判定"] == "是").sum()),
        "threshold_missing": int((s["可做阈值判定"] == "阈值缺失").sum()),
        "threshold_not_applicable": int((s["可做阈值判定"] == "否").sum()),
    }

    # ---- 阈值判定层（3 个可判 + t2n 缺阈值）
    tj = {}
    chem_r = b.chem_restored
    for chem_col, (sens_col, thr, unit, ok) in C.THRESHOLD_JUDGEMENT.items():
        item = {"sensory_col": sens_col, "threshold": thr, "unit": unit, "judgeable": ok}
        if ok:
            oav = (chem_r[chem_col] / thr).dropna()
            item.update({
                "n_valid": int(len(oav)),
                "oav_gt1_n": int((oav > 1).sum()),
                "oav_gt1_ratio": float((oav > 1).mean()),
                "oav_p50": float(oav.median()),
                "oav_max": float(oav.max()),
                "spearman_vs_panel": float(
                    spearmanr(chem_r[chem_col], b.sens[sens_col], nan_policy="omit").statistic),
            })
        tj[chem_col] = item
    rep["threshold_judgement"] = tj

    # ---- 论文划分
    tr, te = b.paper_split()
    style = b.style
    n_all = style.value_counts()
    n_test = style.iloc[te].value_counts().reindex(n_all.index, fill_value=0)
    dev = (n_test - (n_all * C.PAPER_TEST_SIZE).round()).abs()
    rep["paper_split"] = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "test_style_counts": style.iloc[te].value_counts().to_dict(),
        "min_train_class": int(style.iloc[tr].value_counts().min()),
        "max_test_count_dev": int(dev.max()),
        "stratified_ok": bool(dev.max() <= 1),
    }

    # ---- 稳定性折
    folds = {str(seed): [int(len(a)) for a, _ in b.outer_folds(seed)] for seed in C.STABILITY_SEEDS}
    fold_counts = []
    for seed in C.STABILITY_SEEDS:
        for tr_i, te_i in b.outer_folds(seed):
            fold_counts.append(style.iloc[te_i].value_counts().min())
    rep["outer_folds"] = {
        "n_seeds": len(C.STABILITY_SEEDS), "n_folds": C.N_OUTER_FOLDS,
        "fold_sizes": folds[str(C.STABILITY_SEEDS[0])],
        "min_class_in_any_test_fold": int(min(fold_counts)),
        "all_folds_valid": bool(min(fold_counts) >= 1),
    }

    # ---- 写文件
    (C.OUT_DIR / "validation_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
