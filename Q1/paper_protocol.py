"""P-A 论文可比协议（保留候选：ET/RF/HistGB/PLS/Ridge/kNN + 3 基线）。

- 划分：70/30 分层单切（random_state=0）
- 目标：30 个核心 A/F 维
- 管线：折内中位数插补 -> StandardScaler(按需) -> 模型
- 调参：GridSearchCV(cv=5, scoring='r2', 多输出均匀平均)

输出：
  results/paper_protocol.csv
  results/paper_protocol_per_target.csv
  results/paper_test_predictions.csv
被舍弃模型的旧结果见 results/legacy/paper_protocol_legacy_models.csv。
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

import config as C
import dataio
import models

VIEW = "V2_scale_aware"
TARGET_KIND = "core"
INNER_CV = KFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=0)
CANDIDATES = list(C.GRIDS)
BASELINES = ["mean", "style_mean"]


def metrics(y_true, y_pred):
    r2 = np.array([r2_score(y_true[:, j], y_pred[:, j])
                   for j in range(y_true.shape[1])])
    rmse = np.array([np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
                     for j in range(y_true.shape[1])])
    mae = np.array([mean_absolute_error(y_true[:, j], y_pred[:, j])
                    for j in range(y_true.shape[1])])
    return {
        "macro_r2": float(r2.mean()),
        "var_weighted_r2": float(r2_score(y_true, y_pred,
                                          multioutput="variance_weighted")),
        "macro_rmse": float(rmse.mean()),
        "macro_mae": float(mae.mean()),
        "r2": r2, "rmse": rmse, "mae": mae,
    }


def run_model(name, Xtr, ytr, Xte, yte):
    t0 = time.time()
    pipe, grid, inner_jobs = models.build_pipeline(name)
    search = GridSearchCV(pipe, grid, scoring="r2", cv=INNER_CV,
                          n_jobs=inner_jobs, refit=True, error_score="raise")
    search.fit(Xtr, ytr)
    model = search.best_estimator_
    pred = model.predict(Xte)
    m_tr, m_te = metrics(ytr, model.predict(Xtr)), metrics(yte, pred)
    n_feat = Xtr.shape[1]
    row = {
        "model": name,
        "macro_test_r2": m_te["macro_r2"],
        "macro_train_r2": m_tr["macro_r2"],
        "var_weighted_test_r2": m_te["var_weighted_r2"],
        "macro_test_rmse": m_te["macro_rmse"],
        "macro_test_mae": m_te["macro_mae"],
        "n_features": int(n_feat),
        "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=str),
        "fit_seconds": round(time.time() - t0, 1),
    }
    return row, m_te, pred


def run_baselines(Xtr, ytr, Xte, yte, style_tr, style_te):
    rows, details, preds = [], {}, {}

    pred = np.tile(np.asarray(ytr, dtype=float).mean(axis=0), (len(Xte), 1))
    m = metrics(yte, pred)
    rows.append({"model": "mean", "macro_test_r2": m["macro_r2"],
                 "macro_train_r2": 0.0,
                 "var_weighted_test_r2": m["var_weighted_r2"],
                 "macro_test_rmse": m["macro_rmse"],
                 "macro_test_mae": m["macro_mae"],
                 "n_features": 0, "best_params": "{}", "fit_seconds": 0.0})
    details["mean"], preds["mean"] = m, pred

    sm = models.StyleMeanRegressor().fit(Xtr, ytr, style=style_tr)
    pred = sm.predict(Xte, style=style_te)
    m = metrics(yte, pred)
    rows.append({"model": "style_mean", "macro_test_r2": m["macro_r2"],
                 "macro_train_r2": float(metrics(
                     ytr, sm.predict(Xtr, style=style_tr))["macro_r2"]),
                 "var_weighted_test_r2": m["var_weighted_r2"],
                 "macro_test_rmse": m["macro_rmse"],
                 "macro_test_mae": m["macro_mae"],
                 "n_features": 0, "best_params": "{}", "fit_seconds": 0.0})
    details["style_mean"], preds["style_mean"] = m, pred

    return rows, details, preds


def _load_merged():
    res_f = C.OUT_DIR / "paper_protocol.csv"
    per_f = C.OUT_DIR / "paper_protocol_per_target.csv"
    rows = pd.read_csv(res_f).to_dict("records") if res_f.exists() else []
    per = pd.read_csv(per_f).to_dict("records") if per_f.exists() else []
    return rows, per


def _save_merged(rows, per):
    res = pd.DataFrame(rows).drop_duplicates("model", keep="last") \
        .sort_values("macro_test_r2", ascending=False)
    res.to_csv(C.OUT_DIR / "paper_protocol.csv", index=False)
    pd.DataFrame(per).drop_duplicates(["model", "target"], keep="last").to_csv(
        C.OUT_DIR / "paper_protocol_per_target.csv", index=False)
    return res


def _append_model(name, row, m, pred, beer_te):
    rows, per = _load_merged()
    rows.append(row)
    for j, tgt in enumerate(C.CORE_TARGETS):
        per.append({"model": name, "target": tgt,
                    "test_r2": m["r2"][j], "test_rmse": m["rmse"][j],
                    "test_mae": m["mae"][j]})
    pd.DataFrame(pred, columns=C.CORE_TARGETS, index=beer_te) \
        .to_csv(C.OUT_DIR / f"pred_paper_{name}.csv")
    return _save_merged(rows, per)


def _assemble_predictions():
    parts = []
    for f in sorted(C.OUT_DIR.glob("pred_paper_*.csv")):
        name = f.stem.replace("pred_paper_", "")
        d = pd.read_csv(f, index_col=0)
        d.columns = [f"{name}|{c}" for c in d.columns]
        parts.append(d)
    if parts:
        pd.concat(parts, axis=1).sort_index().to_csv(
            C.OUT_DIR / "paper_test_predictions.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔模型名，仅跑这些")
    ap.add_argument("--skip", default="", help="逗号分隔模型名，跳过")
    ap.add_argument("--force", action="store_true", help="忽略已有结果，重跑")
    args = ap.parse_args(argv)

    C.ensure_dirs()
    b = dataio.load_all()
    X, Y = dataio.prepare(b, VIEW, TARGET_KIND)
    tr, te = b.paper_split()
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()
    style_tr, style_te = b.style.iloc[tr].to_numpy(), b.style.iloc[te].to_numpy()
    beers_te = b.beer[te]

    existing, _ = _load_merged()
    done = set() if args.force else {r["model"] for r in existing}
    only = [s for s in args.only.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    names = [n for n in CANDIDATES if (not only or n in only)
             and n not in skip and n not in done]

    for name in names:
        print(f"[{name}] start", flush=True)
        row, m, pred = run_model(name, Xtr, ytr, Xte, yte)
        _append_model(name, row, m, pred, beers_te)
        print(f"[{name}] macro_test_r2={row['macro_test_r2']:.4f} "
              f"train={row['macro_train_r2']:.4f} {row['fit_seconds']}s", flush=True)

    if not only:
        todo = [n for n in BASELINES if n not in done and n not in skip]
        if todo:
            print(f"[baselines] {todo}", flush=True)
            brows, bdetails, bpreds = run_baselines(
                Xtr, ytr, Xte, yte, style_tr, style_te)
            for r in brows:
                if r["model"] in todo:
                    _append_model(r["model"], r, bdetails[r["model"]],
                                  bpreds[r["model"]], beers_te)
                    print(f"[{r['model']}] macro_test_r2={r['macro_test_r2']:.4f}",
                          flush=True)

    _assemble_predictions()
    res = pd.read_csv(C.OUT_DIR / "paper_protocol.csv").sort_values(
        "macro_test_r2", ascending=False)
    print("\n=== ranking ===")
    print(res[["model", "macro_test_r2", "var_weighted_test_r2",
               "macro_train_r2", "fit_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
