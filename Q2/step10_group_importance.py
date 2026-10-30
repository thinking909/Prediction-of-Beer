"""Q2 Step 10: 按化学类别的组级重要性（治共线）。

对每个感官目标，用 Q1 ET 固定参数做 3 折 OOF；把一个化学类别的所有列
作为整体按行置换，看 ΔR² / ΔRMSE。组内共线结构保留，避免单化合物
重要性被共线邻居拆散。随机抽同样多列做对照，给出经验 p 值。

类别由化合物名称规则生成。

输出：results/group/
  q2_group_importance_class.csv
  q2_group_dr2_matrix_class.csv
  q2_group_support_top3.csv
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import q2_config as C
import dataio

N_PERM = 5
N_RAND = 8


def chem_class(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["sulfide", "disulfide", "thiol", "sulfur", "methanethiol"]):
        return "Sulfur"
    if any(k in n for k in ["guaiacol", "phenol", "cresol", "eugenol", "vanillin"]):
        return "Phenol"
    if any(k in n for k in ["acid"]):
        return "Acid"
    if any(k in n for k in [
        "acetate", "butyrate", "hexanoate", "octanoate", "decanoate", "propionate",
        "formate", "lactate", "succinate", "benzoate", "cinnamate", "laurate",
        "myristate", "palmitate", "stearate", "furoate", "tiglate", "valerate",
        "isobutyrate", "isovalerate", "pentanoate", "nicotinate", "salicylate",
        "phenylacetate", "hydrocinnamate", "geranate", "nonanoate", "octadecanoate",
        "hexadecanoate", "tetradecanoate", "pentadecanoate", "decenoate",
    ]):
        return "Ester"
    if any(k in n for k in [
        "linalool", "geraniol", "citronellol", "terpineol", "humulene", "caryophyllene",
        "ocimene", "nerolidol", "farnesene", "farnesol", "terpinen", "limonene",
        "phellandrene", "pinene", "camphor", "ionone", "damascenone", "cadalene",
        "muurolol", "eudesmol", "cadinene", "germacrene", "valencene", "spathulenol",
        "manool", "epicubenol", "cubenol", "nerol", "isoborneol", "terpinyl",
        "myrcene", "cymene", "terpinolene",
    ]):
        return "Terpenoid"
    if any(k in n for k in ["aldehyde"]) or n.endswith("al"):
        return "Aldehyde"
    if n.endswith("one") or "lactone" in n:
        return "Ketone/Lactone"
    if any(k in n for k in ["pyrazine", "furan", "pyrrole", "thiazole", "pyridine"]):
        return "N-heterocycle"
    if n.endswith("ol") or "alcohol" in n:
        return "Alcohol"
    return "Other/Physicochemical"


def fit_folds(X, y, cols, folds):
    fitted = []
    for tr, te in folds:
        Xtr = X.iloc[tr][cols]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        et = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 4})
        et.fit(imp.transform(Xtr), y[tr])
        fitted.append((te, imp, et))
    return fitted


def predict_folds(X, fitted, permute_cols=None, rng=None):
    oof = np.full(len(X), np.nan)
    for te, imp, et in fitted:
        Xte = X.iloc[te].copy()
        if permute_cols:
            idx = rng.permutation(len(te))
            block = Xte[permute_cols].to_numpy()
            Xte[permute_cols] = block[idx]
        oof[te] = et.predict(imp.transform(Xte))
    return oof


def group_importance(X, y, groups, folds, rng, n_perm=N_PERM, n_rand=N_RAND):
    all_cols = list(X.columns)
    fitted = fit_folds(X, y, all_cols, folds)
    base_oof = predict_folds(X, fitted)
    base_r2 = r2_score(y, base_oof)
    base_rmse = float(np.sqrt(mean_squared_error(y, base_oof)))
    out = []
    for g, cols in groups.items():
        if not cols:
            continue
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        perm_r2, perm_rmse = [], []
        for _ in range(n_perm):
            oof = predict_folds(X, fitted, cols, rng)
            perm_r2.append(r2_score(y, oof))
            perm_rmse.append(float(np.sqrt(mean_squared_error(y, oof))))
        obs_dr2 = base_r2 - float(np.mean(perm_r2))
        obs_drmse = (float(np.mean(perm_rmse)) - base_rmse) / max(base_rmse, 1e-9)
        pool = [c for c in all_cols if c not in cols]
        rand_dr2 = []
        for _ in range(n_rand):
            rc = list(rng.choice(pool, size=len(cols), replace=False))
            oof = predict_folds(X, fitted, rc, rng)
            rand_dr2.append(base_r2 - r2_score(y, oof))
        p = (1 + int((np.array(rand_dr2) >= obs_dr2).sum())) / (1 + len(rand_dr2))
        out.append({"group": g, "n_features": len(cols), "delta_r2": obs_dr2,
                    "delta_rmse_pct": obs_drmse, "p_random": p,
                    "rand_dr2_mean": float(np.mean(rand_dr2))})
    d = pd.DataFrame(out).sort_values("delta_r2", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    return d


def main():
    t0 = time.time()
    out = C.OUT_DIR / "group"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    X = b.X(C.VIEW_MAIN)
    klass = {c: chem_class(c) for c in X.columns}
    groups = {}
    for c in X.columns:
        groups.setdefault(klass[c], []).append(c)

    rng = np.random.default_rng(C.SEED)
    rows, top3 = [], []
    final = pd.read_csv(C.OUT_DIR / "final" / "q2_top3_final.csv", encoding="utf-8-sig")
    v3 = final[final.view == "V3"]

    for ti, t in enumerate(C.SENSORY_TARGETS):
        y = b.sens[t].to_numpy()
        folds = list(KFold(3, shuffle=True, random_state=C.SEED + ti).split(X))
        d = group_importance(X, y, groups, folds, rng)
        d.insert(0, "target", t)
        rows.append(d)
        sub = v3[v3.target == t].sort_values("rank")
        for _, r in sub.iterrows():
            gc = klass.get(r.compound, "Unknown")
            g = d[d.group == gc]
            grow = g.iloc[0] if len(g) else None
            top3.append({
                "target": t, "rank": int(r["rank"]), "compound": r.compound,
                "status_final": r.status_final,
                "unit_class": gc,
                "class_rank": int(grow["rank"]) if grow is not None else np.nan,
                "class_delta_r2": float(grow["delta_r2"]) if grow is not None else np.nan,
                "class_p": float(grow["p_random"]) if grow is not None else np.nan,
                # 8 次随机对照的最小 p=0.111，故支持判定用 Top-2 + ΔR²>0.005
                "class_top2": bool(grow is not None and grow["rank"] <= 2 and
                                   grow["delta_r2"] > 0.005),
            })
        print(f"[{ti+1}/50] {t} :: " +
              ", ".join(f"{r.group}({r.delta_r2:+.3f})" for r in d.head(3).itertuples()),
              flush=True)

    cls = pd.concat(rows, ignore_index=True)
    cls.to_csv(out / "q2_group_importance_class.csv", index=False, encoding="utf-8-sig")
    t3 = pd.DataFrame(top3)
    t3.to_csv(out / "q2_group_support_top3.csv", index=False, encoding="utf-8-sig")
    cls.pivot_table(index="target", columns="group", values="delta_r2").to_csv(
        out / "q2_group_dr2_matrix_class.csv", encoding="utf-8-sig")

    stable = t3[t3.status_final == "stable"]
    summ = {
        "seconds": round(time.time() - t0, 1),
        "n_targets": len(C.SENSORY_TARGETS),
        "class_groups": {k: len(v) for k, v in groups.items()},
        "stable_rows": int(len(stable)),
        "stable_class_supported": int(stable.class_top2.sum()),
        "any_class_supported_rows": int(t3.class_top2.sum()),
        "targets_with_class_support": int(t3[t3.class_top2].target.nunique()),
    }
    (out / "step10_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
