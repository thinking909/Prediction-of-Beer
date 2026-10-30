"""线性模型的折内 1%/99% 裁剪对照（报告 3.2 与 3.4 的更正依据）。

- 30 折（10 seeds × 3 折）PLS/Ridge 裁剪后指标 -> linear_clip_stability.csv
- 单切下 V2–V5 的裁剪后指标 -> linear_clip_ablation.csv

未裁剪的 PLS/Ridge 结果只作为反例保留在 step5_summary 里，不再重跑。
"""
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import dataio

INNER_CV = KFold(n_splits=C.N_INNER_FOLDS, shuffle=True, random_state=0)


class ClipQuantile(BaseEstimator, TransformerMixin):
    def __init__(self, q=0.01):
        self.q = q

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.q, axis=0)
        self.hi_ = np.nanquantile(X, 1 - self.q, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _pipe(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", ClipQuantile()),
        ("scale", StandardScaler()),
        ("model", model),
    ])


SPECS = {
    "pls_clip": (lambda: PLSRegression(scale=False),
                 {"model__n_components": [2, 3, 5, 8, 12]}),
    "ridge_clip": (lambda: Ridge(),
                   {"model__alpha": [1.0, 10.0, 100.0]}),
}


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    Y = b.Y("model")
    targets = list(Y.columns)
    core = [i for i, t in enumerate(targets) if t in C.CORE_TARGETS]
    aux = [i for i, t in enumerate(targets) if t in C.AUX_MODEL_TARGETS]

    # 1) 30 折
    rows = []
    for name, (factory, grid) in SPECS.items():
        t0 = time.time()
        for seed in C.STABILITY_SEEDS:
            for k, (tr, te) in enumerate(b.outer_folds(seed)):
                X = b.X("V2_scale_aware")
                gs = GridSearchCV(_pipe(factory()), grid, scoring="r2",
                                  cv=INNER_CV, n_jobs=5, refit=True)
                gs.fit(X.iloc[tr], Y.iloc[tr])
                pred = gs.predict(X.iloc[te])
                yt = Y.iloc[te].to_numpy()
                r = np.array([r2_score(yt[:, j], pred[:, j])
                              for j in range(len(targets))])
                rmse = np.array([np.sqrt(mean_squared_error(yt[:, j], pred[:, j]))
                                 for j in range(len(targets))])
                rows.append({"model": name, "seed": seed, "fold": k,
                             "core_r2": r[core].mean(), "aux_r2": r[aux].mean(),
                             "core_rmse": rmse[core].mean()})
        print(f"[{name}] 30 folds {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "linear_clip_stability.csv", index=False)

    # 2) 单切 × 视图
    tr, te = b.paper_split()
    ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()
    out = []
    for view in ["V2_scale_aware", "V3_absolute", "V4_relative", "V5_restored"]:
        X = b.X(view)
        for name, (factory, grid) in SPECS.items():
            gs = GridSearchCV(_pipe(factory()), grid, scoring="r2",
                              cv=INNER_CV, n_jobs=5, refit=True)
            gs.fit(X.iloc[tr], ytr)
            pred = gs.predict(X.iloc[te])
            r = np.array([r2_score(yte[:, j], pred[:, j])
                          for j in range(len(targets))])
            out.append({"view": view, "model": name, "n_features": X.shape[1],
                        "core_r2": float(r[core].mean()),
                        "aux_r2": float(r[aux].mean()),
                        "best_params": str(gs.best_params_)})
    pd.DataFrame(out).to_csv(C.OUT_DIR / "linear_clip_ablation.csv", index=False)
    print(pd.DataFrame(rows).groupby("model")[["core_r2", "aux_r2"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
