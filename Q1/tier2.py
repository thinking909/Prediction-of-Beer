"""步骤3：Tier-2 逐属性调参（ET / RF）。

对每个感官目标单独做 GridSearchCV(5 折)，两种网格：
  confirm   = config 预注册网格（27 组）
  explore   = config 探索性边界扩展（36 组，明确标注）

与 Tier-1 多输出联合调参（同一 70/30 划分）对比。
输出：
  results/tier2_et_rf.csv        逐维明细
  results/tier2_summary.csv      家族 × 网格 的 macro 汇总
  results/pred_tier2_{fam}_{grid}.csv
"""
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

import config as C
import dataio

VIEW = "V2_scale_aware"
TARGET_KIND = "core"
INNER_CV = KFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=0)


def _make(fam, seed=0):
    if fam == "et":
        return ExtraTreesRegressor(n_estimators=300, bootstrap=True,
                                   random_state=seed, n_jobs=1)
    if fam == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)
    raise ValueError(fam)


def _fit_one(fam, grid, Xtr, Xte, ytr, yte):
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("model", _make(fam))])
    params = {"model__" + k: v for k, v in grid.items()}
    gs = GridSearchCV(pipe, params, scoring="r2", cv=INNER_CV,
                      n_jobs=1, refit=True, error_score="raise")
    gs.fit(Xtr, ytr)
    pred = gs.predict(Xte)
    return {
        "r2": r2_score(yte, pred),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "pred": pred,
        "params": gs.best_params_,
    }


def main():
    fams = [a for a in sys.argv[1:] if a in ("et", "rf")] or ["et", "rf"]
    C.ensure_dirs()
    b = dataio.load_all()
    X, Y = dataio.prepare(b, VIEW, TARGET_KIND)
    tr, te = b.paper_split()
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    beers_te = b.beer[te]

    tier1 = pd.read_csv(C.OUT_DIR / "paper_protocol_per_target.csv")
    t1 = (tier1[tier1.model.isin(fams)]
          .pivot(index="target", columns="model", values="test_r2"))

    rows, preds, summaries = [], {}, []
    for fam in fams:
        grids = {"confirm": C.GRIDS[fam]["param_grid"],
                 "explore": C.EXPLORATORY_GRIDS[fam]}
        for gname, grid in grids.items():
            print(f"[{fam}/{gname}] {int(np.prod([len(v) for v in grid.values()]))} combos x 30 targets",
                  flush=True)
            t0 = time.time()
            res = Parallel(n_jobs=30, prefer="processes")(
                delayed(_fit_one)(fam, grid, Xtr, Xte,
                                  Y.iloc[tr, j].to_numpy(), Y.iloc[te, j].to_numpy())
                for j in range(Y.shape[1]))
            sec = time.time() - t0
            for j, tgt in enumerate(C.CORE_TARGETS):
                r = res[j]
                rows.append({
                    "family": fam, "grid": gname, "target": tgt,
                    "r2_tier1": float(t1.loc[tgt, fam]),
                    "r2_tier2": r["r2"],
                    "delta": r["r2"] - float(t1.loc[tgt, fam]),
                    "rmse_tier2": r["rmse"],
                    "best_params": str(r["params"]),
                })
            preds[(fam, gname)] = pd.DataFrame(
                np.column_stack([r["pred"] for r in res]),
                columns=C.CORE_TARGETS, index=beers_te)
            summaries.append({
                "family": fam, "grid": gname,
                "macro_r2_tier1": float(t1[fam].mean()),
                "macro_r2_tier2": float(np.mean([r["r2"] for r in res])),
                "seconds": round(sec, 1),
            })
            print(f"[{fam}/{gname}] macro_tier2="
                  f"{summaries[-1]['macro_r2_tier2']:.4f} "
                  f"(tier1={summaries[-1]['macro_r2_tier1']:.4f}) {sec:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(C.OUT_DIR / "tier2_et_rf.csv", index=False)
            pd.DataFrame(summaries).to_csv(C.OUT_DIR / "tier2_summary.csv", index=False)

    for (fam, gname), df in preds.items():
        df.to_csv(C.OUT_DIR / f"pred_tier2_{fam}_{gname}.csv")
    shared_params_check(fams, Xtr, Xte, Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy(), t1)
    print("\n=== summary ===")
    print(pd.DataFrame(summaries).to_string(index=False))


def shared_params_check(fams, Xtr, Xte, ytr, yte, t1):
    """逐维独立拟合、但沿用 Tier-1 共享超参（报告 2.3 的对照）。"""
    import json
    from joblib import Parallel, delayed
    from sklearn.base import clone
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import r2_score

    res_f = C.OUT_DIR / "paper_protocol.csv"
    if not res_f.exists():
        print("[shared-params] skip: paper_protocol.csv 不存在", flush=True)
        return
    tier1 = pd.read_csv(res_f).set_index("model")
    imp = SimpleImputer(strategy="median").fit(Xtr)
    A, B = imp.transform(Xtr), imp.transform(Xte)
    rows = []
    for fam in fams:
        if fam not in tier1.index:
            continue
        params = {k.replace("model__", ""): v
                  for k, v in json.loads(tier1.loc[fam, "best_params"]).items()}
        base = (ExtraTreesRegressor(n_estimators=300, bootstrap=True,
                                    random_state=0, n_jobs=1) if fam == "et"
                else RandomForestRegressor(n_estimators=300, random_state=0,
                                           n_jobs=1))
        est = clone(base).set_params(**params)

        def one(j):
            m = clone(est).fit(A, ytr[:, j])
            return r2_score(yte[:, j], m.predict(B))

        r2 = Parallel(n_jobs=30)(delayed(one)(j) for j in range(ytr.shape[1]))
        for j, tgt in enumerate(C.CORE_TARGETS):
            rows.append({"family": fam, "target": tgt,
                         "r2_multio": float(t1.loc[tgt, fam]),
                         "r2_indep_shared": float(r2[j])})
        print(f"[shared-params/{fam}] macro="
              f"{np.mean([r2[j] for j in range(len(C.CORE_TARGETS))]):.4f}", flush=True)
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "tier2_shared_params_check.csv",
                              index=False)


if __name__ == "__main__":
    main()
