"""步骤6：P-C 交付协议。

- 用 P-B 胜者 ET 的外折预测（seed 20261023 的 3 折 OOF）
- 折外等张校准（leave-fold-out isotonic）
- 观测 1/7..6/7 分位阈值 -> 相对 0-6 等级
- 三种阈值方案敏感性（quantile / kmeans / equal_freq）
- 4 个缺陷维的浓度+阈值判定（t2n 标注不可判定）
- 全量 ET 模型与阈值序列化（部署用，不参与评估）

输出：
  results/relative_scores_v3.csv
  results/oof_calibrated.csv
  results/level_distribution.csv
  results/mapping_sensitivity.csv
  results/per_target_calibration.csv
  results/fault_threshold_flags.csv
  results/mapping_rules_v3.json
  results/final_model.joblib
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score

import config as C
import dataio
import models

warnings.filterwarnings("ignore")
SEED = C.STABILITY_SEEDS[0]
LEVELS = np.arange(1, C.N_LEVELS) / C.N_LEVELS   # 1/7 .. 6/7


def assign(pred, thresholds):
    lv = np.searchsorted(thresholds, pred, side="right")
    return np.clip(lv, 0, C.N_LEVELS - 1)


def levels_kmeans(pred):
    km = KMeans(n_clusters=C.N_LEVELS, n_init=10, random_state=0)
    lab = km.fit_predict(pred.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())
    rank = {c: i for i, c in enumerate(order)}
    return np.array([rank[l] for l in lab])


def level_stats(levels):
    counts = np.bincount(levels, minlength=C.N_LEVELS)
    share = counts / counts.sum()
    return {
        "empty_levels": int((counts == 0).sum()),
        "mid_share": float(share[3] + share[4]),
        "max_level_share": float(share.max()),
        "n_levels_used": int((counts > 0).sum()),
    }


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    X, Y = dataio.prepare(b, "V2_scale_aware", "model")
    targets = list(Y.columns)
    folds = b.outer_folds(SEED)

    oof = pd.read_csv(C.OUT_DIR / f"oof_et_{SEED}.csv", index_col=0)
    oof = oof.loc[b.beer, targets]
    assert oof.notna().all().all(), "OOF 存在缺失"

    # ---------- 折外等张校准 ----------
    cal = pd.DataFrame(index=Y.index, columns=targets, dtype=float)
    cal_models = {}
    for k, (tr, te) in enumerate(folds):
        other = np.setdiff1d(np.arange(len(Y)), te)
        for tgt in targets:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof.iloc[other][tgt].to_numpy(), Y.iloc[other][tgt].to_numpy())
            cal.iloc[te, cal.columns.get_loc(tgt)] = iso.predict(
                oof.iloc[te][tgt].to_numpy())
            cal_models.setdefault(tgt, []).append(iso)
    cal = cal.astype(float)

    # ---------- 阈值与等级 ----------
    obs_thr = {t: np.quantile(Y[t].to_numpy(), LEVELS) for t in targets}
    eqf_thr = {t: np.quantile(cal[t].to_numpy(), LEVELS) for t in targets}

    levels = {s: pd.DataFrame(index=Y.index, columns=targets, dtype=int)
              for s in C.THRESHOLD_SCHEMES}
    raw_levels = pd.DataFrame(index=Y.index, columns=targets, dtype=int)
    for t in targets:
        levels["quantile"][t] = assign(cal[t].to_numpy(), obs_thr[t])
        levels["equal_freq"][t] = assign(cal[t].to_numpy(), eqf_thr[t])
        levels["kmeans"][t] = levels_kmeans(cal[t].to_numpy())
        raw_levels[t] = assign(oof[t].to_numpy(), obs_thr[t])

    # ---------- 逐维诊断 ----------
    rows, dist = [], []
    for t in targets:
        sp_raw = spearmanr(oof[t], Y[t]).statistic
        sp_cal = spearmanr(cal[t], Y[t]).statistic
        st_q = level_stats(levels["quantile"][t].to_numpy())
        st_raw = level_stats(raw_levels[t].to_numpy())
        kap = {}
        for a, c in [("quantile", "kmeans"), ("quantile", "equal_freq"),
                     ("kmeans", "equal_freq")]:
            try:
                kap[f"kappa_{a}_vs_{c}"] = float(cohen_kappa_score(
                    levels[a][t], levels[c][t], weights="quadratic"))
            except ValueError:
                kap[f"kappa_{a}_vs_{c}"] = np.nan
        rows.append({
            "target": t,
            "spearman_raw": float(sp_raw),
            "spearman_calibrated": float(sp_cal),
            "cal_gain": float(sp_cal - sp_raw),
            "empty_levels_cal": st_q["empty_levels"],
            "empty_levels_raw": st_raw["empty_levels"],
            "mid_share_cal": st_q["mid_share"],
            "mid_share_raw": st_raw["mid_share"],
            "levels_used_cal": st_q["n_levels_used"],
            **kap,
        })
        for s in C.THRESHOLD_SCHEMES:
            cnt = np.bincount(levels[s][t].to_numpy(), minlength=C.N_LEVELS)
            for lv in range(C.N_LEVELS):
                dist.append({"target": t, "scheme": s, "level": lv, "n": int(cnt[lv])})
    diag = pd.DataFrame(rows)
    diag.to_csv(C.OUT_DIR / "per_target_calibration.csv", index=False)
    diag[["target", "kappa_quantile_vs_kmeans", "kappa_quantile_vs_equal_freq",
          "kappa_kmeans_vs_equal_freq"]].to_csv(
        C.OUT_DIR / "mapping_sensitivity.csv", index=False)
    pd.DataFrame(dist).to_csv(C.OUT_DIR / "level_distribution.csv", index=False)

    # ---------- 交付表 ----------
    out = pd.concat({"level": levels["quantile"],
                     "calibrated_score": cal.round(6)}, axis=1)
    out.columns = [f"{kind}|{t}" for kind, t in out.columns]
    out.index.name = "beer"
    out.to_csv(C.OUT_DIR / "relative_scores_v3.csv")
    cal.to_csv(C.OUT_DIR / "oof_calibrated.csv")

    # ---------- 缺陷阈值判定 ----------
    flags = []
    for chem_col, (sens_col, thr, unit, ok) in C.THRESHOLD_JUDGEMENT.items():
        conc = b.chem_restored[chem_col]
        if ok:
            oav = conc / thr
            lv = pd.cut(oav, [-np.inf, 1, 2, 5, np.inf],
                        labels=["below", "slight", "moderate", "strong"])
            lv = lv.astype(object).where(conc.notna(), "no_measurement")
            for beer, val in lv.items():
                flags.append({"beer": beer, "chemical": chem_col,
                              "sensory_target": sens_col, "threshold": thr,
                              "unit": unit, "concentration": conc.loc[beer],
                              "oav": oav.loc[beer], "flag": val,
                              "judgeable": True})
        else:
            for beer in b.beer:
                flags.append({"beer": beer, "chemical": chem_col,
                              "sensory_target": sens_col, "threshold": np.nan,
                              "unit": unit, "concentration": conc.loc[beer],
                              "oav": np.nan, "flag": "unjudgeable",
                              "judgeable": False})
    pd.DataFrame(flags).to_csv(C.OUT_DIR / "fault_threshold_flags.csv", index=False)

    # ---------- 全量模型与规则（部署） ----------
    imp = SimpleImputer(strategy="median").fit(X)
    final = models.final_et()
    final.fit(imp.transform(X), Y.to_numpy())
    joblib.dump({
        "imputer": imp, "model": final, "features": list(X.columns),
        "targets": targets, "calibrators": cal_models,
        "thresholds": {t: obs_thr[t].tolist() for t in targets},
        "view": "V2_scale_aware",
    }, C.OUT_DIR / "final_model.joblib")

    rules = {
        "level_definition": "0=观测分布最低 1/7 段，6=最高 1/7 段；不是原始人工 0-6 分",
        "thresholds": {t: obs_thr[t].tolist() for t in targets},
        "calibration": "leave-fold-out isotonic on 3-fold OOF (seed 20261023)",
        "oof_source": f"oof_et_{SEED}.csv",
        "threshold_schemes": C.THRESHOLD_SCHEMES,
        "limitations": [
            "相对等级不等于原始人工 0-6 评分，禁止报告伪准确率/Kappa",
            "等级依赖观测标签分布；若真实分布非近似均匀，边缘档最多错位 1 档",
            "t2n (E-2-nonenal) 无文献阈值，只能给相对分位，不能做绝对判定",
            "DMS/Diacetyl/4-VG 的阈值判定与感官预测是两套结论，不互相验证",
        ],
    }
    (C.OUT_DIR / "mapping_rules_v3.json").write_text(
        json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    print(diag[["spearman_raw", "spearman_calibrated", "cal_gain",
                "empty_levels_cal", "mid_share_cal"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
