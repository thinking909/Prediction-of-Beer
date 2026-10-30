"""步骤4：尺度视图消融 V1–V5。

固定 70/30 划分与调参协议，比较：
  V1 全部 231（含 4 派生）
  V2 227（去派生，主视图）
  V3 仅绝对 70
  V4 仅相对 157
  V5 227 还原浓度（V2 的单调变换）

模型：ET（树，主）、PLS / Ridge / kNN（线性或近邻对照）。
目标：30 核心 + 16 辅助（4 个阈值维不建模），两组分别汇总。

输出：
  results/ablation_scale_views.csv
  results/ablation_per_target.csv
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

import config as C
import dataio
import models

INNER_CV = KFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=0)
VIEWS = ["V1_paper_all", "V2_scale_aware", "V3_absolute",
         "V4_relative", "V5_restored"]
MODELS = ["et", "pls", "ridge", "knn"]


def build(view, model):
    pipe, grid, inner_jobs = models.build_pipeline(model)
    return pipe, grid, inner_jobs


def block_metrics(y_true, y_pred, cols):
    r2 = np.array([r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])])
    rmse = np.array([np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
                     for j in range(y_true.shape[1])])
    return r2, rmse


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    Xall = {v: b.X(v) for v in VIEWS}
    Y = b.Y("model")
    tr, te = b.paper_split()
    targets = list(Y.columns)
    core_idx = [i for i, t in enumerate(targets) if t in C.CORE_TARGETS]
    aux_idx = [i for i, t in enumerate(targets) if t in C.AUX_MODEL_TARGETS]
    ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()

    rows, per_target, preds = [], [], {}
    for view in VIEWS:
        Xtr, Xte = Xall[view].iloc[tr], Xall[view].iloc[te]
        for model in MODELS:
            t0 = time.time()
            pipe, grid, gj = build(view, model)
            gs = GridSearchCV(pipe, grid, scoring="r2", cv=INNER_CV,
                              n_jobs=gj, refit=True,
                              error_score="raise")
            gs.fit(Xtr, ytr)
            pred = gs.predict(Xte)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            r2, rmse = block_metrics(yte, pred, targets)
            sec = time.time() - t0
            preds[(view, model)] = pred
            for name, idx in (("core", core_idx), ("aux", aux_idx)):
                rows.append({
                    "view": view, "model": model, "targets": name,
                    "n_features": Xtr.shape[1],
                    "macro_r2": float(r2[idx].mean()),
                    "macro_rmse": float(rmse[idx].mean()),
                    "best_params": str(gs.best_params_) if grid else "{}",
                    "seconds": round(sec, 1),
                })
            for j, tgt in enumerate(targets):
                per_target.append({
                    "view": view, "model": model, "target": tgt,
                    "kind": "core" if j in core_idx else "aux",
                    "test_r2": r2[j], "test_rmse": rmse[j],
                })
            print(f"[{view}/{model}] core={rows[-2]['macro_r2']:.4f} "
                  f"aux={rows[-1]['macro_r2']:.4f} {sec:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(C.OUT_DIR / "ablation_scale_views.csv", index=False)
            pd.DataFrame(per_target).to_csv(C.OUT_DIR / "ablation_per_target.csv", index=False)

    # 单调变换健全性检查：树模型在 V2 与 V5 上预测应一致
    for model in ("et",):
        d = np.abs(preds[("V2_scale_aware", model)] - preds[("V5_restored", model)])
        print(f"[note] {model} V2 vs V5 max|Δpred| = {d.max():.3e} (sklearn 阈值非秩次，非严格单调不变)", flush=True)

    # 逐维对照：绝对/相对/差
    pt = pd.DataFrame(per_target)
    wide = pt.pivot_table(index="target", columns="view", values="test_r2")
    wide["abs_minus_rel"] = wide["V3_absolute"] - wide["V4_relative"]
    wide.to_csv(C.OUT_DIR / "ablation_role_by_target.csv")


if __name__ == "__main__":
    main()
