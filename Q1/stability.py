"""步骤5：P-B 稳定性协议。

- 10 seeds × 3 折风格分层外折
- 候选内层 GridSearchCV(cv=5)，嵌套在每个外折内完成
- 主指标：core/aux macro R²、macro RMSE
- 逐 seed 配对 Wilcoxon + 符号计数（≥8/10 才判可分）
- ET 逐维 R² 95% 区间与强度分级

另附：固定参数的尺度视图稳定性检查（V2/V3/V4/V5），复核步骤4。

输出：
  results/stability_folds.csv
  results/stability_summary.csv
  results/per_seed_delta.csv
  results/paired_tests.csv
  results/per_target_ci.csv
  results/view_stability.csv
"""
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

import config as C
import dataio
import models

warnings.filterwarnings("ignore")
INNER_CV = KFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=0)
VIEW = "V2_scale_aware"
Y = None


def metrics_block(y_true, y_pred, core_idx, aux_idx):
    r2 = np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])])
    rmse = np.array([np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
                     for j in range(y_true.shape[1])])
    mae = np.array([mean_absolute_error(y_true[:, j], y_pred[:, j])
                    for j in range(y_true.shape[1])])
    return {
        "core_r2": float(r2[core_idx].mean()),
        "aux_r2": float(r2[aux_idx].mean()),
        "all_r2": float(r2.mean()),
        "core_rmse": float(rmse[core_idx].mean()),
        "aux_rmse": float(rmse[aux_idx].mean()),
        "r2": r2, "rmse": rmse, "mae": mae,
    }


def main():
    global Y
    C.ensure_dirs()
    b = dataio.load_all()
    Y = b.Y("model")
    targets = list(Y.columns)
    core_idx = [i for i, t in enumerate(targets) if t in C.CORE_TARGETS]
    aux_idx = [i for i, t in enumerate(targets) if t in C.AUX_MODEL_TARGETS]
    X = b.X(VIEW)
    style = b.style.to_numpy()

    candidates = ["et", "rf", "hist_gb", "pls", "ridge", "knn", "style_mean", "mean"]
    fold_rows, target_rows = [], []
    oof = {}
    t_all = time.time()

    for seed in C.STABILITY_SEEDS:
        t_seed = time.time()
        folds = b.outer_folds(seed)
        for k, (tr, te) in enumerate(folds):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()
            for name in candidates:
                t0 = time.time()
                if name == "style_mean":
                    m = models.StyleMeanRegressor().fit(Xtr, ytr, style=style[tr])
                    pred = m.predict(Xte, style=style[te])
                    params = "{}"
                elif name == "mean":
                    pred = np.tile(ytr.mean(0), (len(te), 1))
                    params = "{}"
                else:
                    pipe, grid, gj = models.build_pipeline(name)
                    gs = GridSearchCV(pipe, grid, scoring="r2", cv=INNER_CV,
                                      n_jobs=gj, refit=True, error_score="raise")
                    gs.fit(Xtr, ytr)
                    pred = gs.predict(Xte)
                    params = str(gs.best_params_)
                    if name == "et":
                        oof.setdefault(seed, np.full((len(Y), len(targets)), np.nan))
                        oof[seed][te] = pred
                mb = metrics_block(yte, pred, core_idx, aux_idx)
                fold_rows.append({
                    "seed": seed, "fold": k, "model": name,
                    "core_r2": mb["core_r2"], "aux_r2": mb["aux_r2"],
                    "all_r2": mb["all_r2"], "core_rmse": mb["core_rmse"],
                    "aux_rmse": mb["aux_rmse"], "best_params": params,
                    "seconds": round(time.time() - t0, 1),
                })
                if name == "et":
                    for j, tgt in enumerate(targets):
                        target_rows.append({"seed": seed, "fold": k, "target": tgt,
                                            "kind": "core" if j in core_idx else "aux",
                                            "r2": mb["r2"][j], "rmse": mb["rmse"][j]})
            pd.DataFrame(fold_rows).to_csv(C.OUT_DIR / "stability_folds.csv", index=False)
            pd.DataFrame(target_rows).to_csv(C.OUT_DIR / "per_target_folds.csv", index=False)
        print(f"[seed {seed}] {time.time()-t_seed:.0f}s", flush=True)

    print(f"primary done in {(time.time()-t_all)/60:.1f} min", flush=True)

    # 尺度视图固定参数复核
    view_rows = []
    for view in ["V2_scale_aware", "V3_absolute", "V4_relative", "V5_restored"]:
        Xv = b.X(view)
        for seed in C.STABILITY_SEEDS:
            for k, (tr, te) in enumerate(b.outer_folds(seed)):
                imp = SimpleImputer(strategy="median").fit(Xv.iloc[tr])
                m = models.final_et().fit(imp.transform(Xv.iloc[tr]),
                                   Y.iloc[tr].to_numpy())
                pred = m.predict(imp.transform(Xv.iloc[te]))
                mb = metrics_block(Y.iloc[te].to_numpy(), pred, core_idx, aux_idx)
                view_rows.append({"view": view, "seed": seed, "fold": k,
                                  "core_r2": mb["core_r2"], "aux_r2": mb["aux_r2"],
                                  "core_rmse": mb["core_rmse"]})
        print(f"[view {view}] done", flush=True)
    pd.DataFrame(view_rows).to_csv(C.OUT_DIR / "view_stability.csv", index=False)

    # OOF（seed 20261023）供步骤6 参考
    s0 = C.STABILITY_SEEDS[0]
    pd.DataFrame(oof[s0], columns=targets, index=b.beer).to_csv(
        C.OUT_DIR / f"oof_et_{s0}.csv")

    aggregate(pd.DataFrame(fold_rows), pd.DataFrame(target_rows),
              pd.DataFrame(view_rows), targets)


def aggregate(folds, targets_df, views, targets):
    # 逐模型汇总
    summ = []
    for name, d in folds.groupby("model"):
        per_seed = d.groupby("seed")[["core_r2", "aux_r2", "core_rmse", "aux_rmse"]].mean()
        summ.append({
            "model": name,
            "core_r2_mean": per_seed.core_r2.mean(),
            "core_r2_sd": per_seed.core_r2.std(ddof=1),
            "aux_r2_mean": per_seed.aux_r2.mean(),
            "aux_r2_sd": per_seed.aux_r2.std(ddof=1),
            "core_rmse_mean": per_seed.core_rmse.mean(),
            "core_rmse_sd": per_seed.core_rmse.std(ddof=1),
            "seconds_per_fold": d.seconds.mean(),
        })
    summ = pd.DataFrame(summ).sort_values("core_r2_mean", ascending=False)
    summ.to_csv(C.OUT_DIR / "stability_summary.csv", index=False)

    # 逐 seed 配对（ET 为参照）
    per_seed = folds.groupby(["seed", "model"])[["core_rmse", "core_r2"]].mean().reset_index()
    deltas = []
    for name in summ.model:
        if name == "et":
            continue
        a = per_seed[per_seed.model == "et"].set_index("seed").core_rmse
        c = per_seed[per_seed.model == name].set_index("seed").core_rmse
        d = (c - a).reindex(a.index)   # >0 表示 ET 更好（RMSE 更低）
        try:
            p = wilcoxon(a, c, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        deltas.append({"comparison": f"et_vs_{name}", "mean_delta_rmse": d.mean(),
                       "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                       "wilcoxon_p": p,
                       "separable": bool((d > 0).sum() >= 8 and d.mean() > 0)})
        for s, v in d.items():
            deltas.append({"comparison": f"et_vs_{name}|seed{s}", "mean_delta_rmse": v})
    pd.DataFrame(deltas).to_csv(C.OUT_DIR / "per_seed_delta.csv", index=False)
    pt = pd.DataFrame([r for r in deltas if "|seed" not in r["comparison"]])
    pt.to_csv(C.OUT_DIR / "paired_tests.csv", index=False)

    # 逐维 CI（ET）
    ci = []
    et = targets_df[targets_df.seed.isin(C.STABILITY_SEEDS)]
    for tgt, d in et.groupby("target"):
        r = d.r2.to_numpy()
        mean, lo, hi = r.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)
        if mean >= C.R2_STRONG and lo > 0.20:
            grade = "strong"
        elif mean >= C.R2_MEDIUM:
            grade = "medium"
        else:
            grade = "weak"
        ci.append({"target": tgt, "kind": d.kind.iloc[0], "mean_r2": mean,
                   "p2.5": lo, "p97.5": hi, "sd": r.std(ddof=1), "grade": grade})
    pd.DataFrame(ci).sort_values("mean_r2", ascending=False).to_csv(
        C.OUT_DIR / "per_target_ci.csv", index=False)

    # 视图复核
    vs = []
    for view, d in views.groupby("view"):
        per_seed = d.groupby("seed")[["core_r2", "aux_r2", "core_rmse"]].mean()
        vs.append({"view": view, "core_r2_mean": per_seed.core_r2.mean(),
                   "core_r2_sd": per_seed.core_r2.std(ddof=1),
                   "aux_r2_mean": per_seed.aux_r2.mean(),
                   "aux_r2_sd": per_seed.aux_r2.std(ddof=1),
                   "core_rmse_mean": per_seed.core_rmse.mean()})
    vs = pd.DataFrame(vs).sort_values("core_r2_mean", ascending=False)
    vs.to_csv(C.OUT_DIR / "view_stability_summary.csv", index=False)



if __name__ == "__main__":
    main()
