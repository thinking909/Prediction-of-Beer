"""尺度消融的对照实验（报告 3.4 引用）。

1. 随机 70 列子集：仅相对 157 与混合 227 各抽 10 次（固定 ET 参数）
2. 单调变换检查：同一插补值下，V2 与还原浓度 V5 的预测差异（RF/ET）

输出：results/ablation_controls.json
"""
import json

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

import config as C
import dataio
import models


def pred_fixed(X, ytr, tr, te, est):
    imp = SimpleImputer(strategy="median").fit(X.iloc[tr])
    est.fit(imp.transform(X.iloc[tr]), ytr)
    return est.predict(imp.transform(X.iloc[te]))


def main():
    C.ensure_dirs()
    b = dataio.load_all()
    Y = b.Y("model")
    tr, te = b.paper_split()
    ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()
    targets = list(Y.columns)
    core = [i for i, t in enumerate(targets) if t in C.CORE_TARGETS]
    aux = [i for i, t in enumerate(targets) if t in C.AUX_MODEL_TARGETS]

    def et():
        return models.final_et()

    res = {}
    for name, view, seed in [("random70_of_V4", "V4_relative", 0),
                             ("random70_of_V2", "V2_scale_aware", 1)]:
        X = b.X(view)
        rng = np.random.default_rng(seed)
        cols = list(X.columns)
        draws = []
        for _ in range(10):
            c = list(rng.choice(cols, 70, replace=False))
            p = pred_fixed(X[c], ytr, tr, te, et())
            draws.append({
                "core": float(np.mean([r2_score(yte[:, j], p[:, j]) for j in core])),
                "aux": float(np.mean([r2_score(yte[:, j], p[:, j]) for j in aux])),
            })
        res[name] = {
            "mean_core": float(np.mean([d["core"] for d in draws])),
            "min_core": float(np.min([d["core"] for d in draws])),
            "max_core": float(np.max([d["core"] for d in draws])),
            "mean_aux": float(np.mean([d["aux"] for d in draws])),
            "draws": draws,
        }
        print(f"[{name}] core={res[name]['mean_core']:.4f} "
              f"({res[name]['min_core']:.4f}–{res[name]['max_core']:.4f})", flush=True)

    # 单调变换：同一插补值下 V2 vs V5
    X2 = b.X("V2_scale_aware")
    scale = b.scale.set_index("列名")
    log_cols = [c for c in X2.columns
                if scale.loc[c, "尺度类别"] in ("log10还原", "未校准(log10还原)")]
    pos = [X2.columns.get_loc(c) for c in log_cols]
    imp = SimpleImputer(strategy="median").fit(X2.iloc[tr])
    A, B = imp.transform(X2.iloc[tr]), imp.transform(X2.iloc[te])
    A5, B5 = A.copy(), B.copy()
    A5[:, pos] = 10 ** A5[:, pos]
    B5[:, pos] = 10 ** B5[:, pos]

    def fit(est, A, B):
        est.fit(A, ytr)
        return est.predict(B)

    res["invariance_maxdiff"] = {}
    for name, est, kw in [
            ("rf", RandomForestRegressor(n_estimators=300, max_features=0.6,
                                         max_depth=None, min_impurity_decrease=0.0,
                                         random_state=0, n_jobs=30), {}),
            ("et", ExtraTreesRegressor(n_estimators=300, bootstrap=True,
                                       max_features=0.6, max_depth=None,
                                       min_impurity_decrease=0.0,
                                       random_state=0, n_jobs=30), {})]:
        p = fit(est, A, B)
        q = fit(est, A5, B5)
        res["invariance_maxdiff"][name] = float(np.abs(p - q).max())

    (C.OUT_DIR / "ablation_controls.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(res["invariance_maxdiff"])


if __name__ == "__main__":
    main()
