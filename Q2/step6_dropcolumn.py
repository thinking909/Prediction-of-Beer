"""Q2 Step 6：drop-column 预测贡献。

对 stage1 Top-3：
- 用 Q1 ET 固定参数、3 折 OOF；
- 基线、单删每个 Top-3、整组删除；
- 整组 ΔR²>0.01 或 ΔRMSE>2% 才保留 stable_candidate。
输出：Q2/results/dropcolumn/*
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

EXAMPLES = ["Acidity", "Hops aroma - overall", "Esters aroma - floral", "Dimethyl sulfide"]


def evaluate(X, y, drop_cols, folds):
    oof = np.full(len(y), np.nan)
    cols = [c for c in X.columns if c not in drop_cols]
    for tr, te in folds:
        Xtr, Xte = X.iloc[tr][cols], X.iloc[te][cols]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        et = ExtraTreesRegressor(**{**C.q1c.FINAL_ET_PARAMS, "n_jobs": 1})
        et.fit(imp.transform(Xtr), y[tr])
        oof[te] = et.predict(imp.transform(Xte))
    return {
        "r2": r2_score(y, oof),
        "rmse": float(np.sqrt(mean_squared_error(y, oof))),
    }


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "dropcolumn"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    stage1 = pd.read_csv(C.OUT_DIR / "consensus" / "q2_top3_stage1.csv")
    rows, stage2 = [], []
    t0 = time.time()
    for vi, (view, key) in enumerate([("V3", C.VIEW_MAIN), ("V2", C.VIEW_CHECK)]):
        X = b.X(key)
        print(f"[{view}] start", flush=True)
        for ti, t in enumerate(C.SENSORY_TARGETS):
            d = stage1[(stage1.view == view) & (stage1.target == t)]
            top = list(d.sort_values("rank").compound)
            if not top:
                continue
            y = b.sens[t].to_numpy()
            folds = list(KFold(3, shuffle=True,
                               random_state=C.SEED + 100 * vi + ti).split(X))
            base = evaluate(X, y, [], folds)
            for c in top:
                r = evaluate(X, y, [c], folds)
                rows.append({
                    "view": view, "target": t, "compound": c, "drop_type": "single",
                    "r2_base": base["r2"], "r2_drop": r["r2"],
                    "delta_r2": base["r2"] - r["r2"],
                    "rmse_base": base["rmse"], "rmse_drop": r["rmse"],
                    "delta_rmse_pct": (r["rmse"] - base["rmse"]) / max(base["rmse"], 1e-9),
                })
            g = evaluate(X, y, top, folds)
            gd_r2 = base["r2"] - g["r2"]
            gd_rmse = (g["rmse"] - base["rmse"]) / max(base["rmse"], 1e-9)
            rows.append({
                "view": view, "target": t, "compound": "|".join(top),
                "drop_type": "group",
                "r2_base": base["r2"], "r2_drop": g["r2"],
                "delta_r2": gd_r2,
                "rmse_base": base["rmse"], "rmse_drop": g["rmse"],
                "delta_rmse_pct": gd_rmse,
            })
            weak = base["r2"] < C.q1c.R2_MEDIUM
            passed = bool(gd_rmse > C.DROP_RMSE_PCT) if weak else bool(
                gd_r2 > C.DROP_R2 or gd_rmse > C.DROP_RMSE_PCT)
            for _, r in d.sort_values("rank").iterrows():
                status2 = r.status_stage1
                if status2 == "stable_candidate" and not passed:
                    status2 = "exploratory"
                stage2.append({
                    **r.to_dict(),
                    "delta_r2_group": gd_r2,
                    "delta_rmse_group_pct": gd_rmse,
                    "baseline_weak": weak,
                    "drop_pass": passed,
                    "status_stage2": status2,
                })
        print(f"[{view}] done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out / "q2_dropcolumn.csv", index=False)
    s2 = pd.DataFrame(stage2)
    s2.to_csv(out / "q2_top3_stage2.csv", index=False)

    summary = {
        "seconds": round(time.time() - t0, 1),
        "stage1_stable_rows": {
            v: int((stage1[stage1.view == v].status_stage1 == "stable_candidate").sum())
            for v in ["V3", "V2"]
        },
        "stage2_stable_rows": {
            v: int((s2[s2.view == v].status_stage2 == "stable_candidate").sum())
            for v in ["V3", "V2"]
        },
        "downgraded_rows": {
            v: int(((s2.view == v) & (s2.status_stage1 == "stable_candidate") &
                    (s2.status_stage2 == "exploratory")).sum())
            for v in ["V3", "V2"]
        },
        "targets_drop_pass": {
            v: int(s2[(s2.view == v) & s2.drop_pass].target.nunique())
            for v in ["V3", "V2"]
        },
    }
    # 例子
    ex = {}
    for view in ["V3", "V2"]:
        for t in EXAMPLES:
            g = d[(d.view == view) & (d.target == t) & (d.drop_type == "group")]
            if len(g):
                row = g.iloc[0]
                ex[f"{view}|{t}"] = {
                    c: float(row[c]) for c in
                    ["r2_base", "r2_drop", "delta_r2", "delta_rmse_pct"]
                }
    summary["examples"] = ex
    (out / "step6_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
