"""参考原论文方法做的移植试验（仅用附件数据，不用外部标签）。

试验：
A. 50 维全模型 vs 当前 46 维（ET 最终参数）
B. 论文风格 ET 网格（max_features 绝对比例、max_depth 10–18、min_impurity 0–0.3、n_est 500）
C. mean vs median 插补

只在 70/30 单切上比较，用于判断是否值得改主流程。
输出：results/paper_methods_test.csv
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold

import config as C
import dataio

VIEW = "V2_scale_aware"
INNER = KFold(5, shuffle=True, random_state=0)
CORE = list(C.CORE_TARGETS)
AUX = list(C.AUX_MODEL_TARGETS)
DEFECT = list(C.THRESHOLD_TARGETS)


def et(params=None):
    p = dict(n_estimators=300, bootstrap=True, max_features=0.6, max_depth=None,
             min_impurity_decrease=0.0, random_state=0, n_jobs=30)
    if params:
        p.update(params)
    return ExtraTreesRegressor(**p)


def eval_pred(y, pred, cols):
    r2 = np.array([r2_score(y[:, j], pred[:, j]) for j in range(y.shape[1])])
    sets = {"core": CORE, "aux": AUX, "defect": DEFECT}
    idx = {k: [i for i, t in enumerate(cols) if t in names]
           for k, names in sets.items()}
    return {
        "core": float(r2[idx["core"]].mean()),
        "aux": float(r2[idx["aux"]].mean()),
        "defect": float(r2[idx["defect"]].mean()),
        "all": float(r2.mean()),
        "var_weighted": float(r2_score(y, pred, multioutput="variance_weighted")),
    }


def run(y_cols, params=None, strategy="median", tune=False):
    b = dataio.load_all()
    X = b.X(VIEW)
    Y = b.sens[y_cols]
    tr, te = b.paper_split()
    imp = SimpleImputer(strategy=strategy).fit(X.iloc[tr])
    A, B = imp.transform(X.iloc[tr]), imp.transform(X.iloc[te])
    ytr, yte = Y.iloc[tr].to_numpy(), Y.iloc[te].to_numpy()
    if tune:
        grid = {
            "max_features": [0.2, 0.4, 0.6, 0.8],
            "max_depth": [10, 18, None],
            "min_impurity_decrease": [0.0, 0.1, 0.2, 0.3],
            "n_estimators": [500],
        }
        gs = GridSearchCV(et(), grid, scoring="r2", cv=INNER, n_jobs=1,
                          refit=True)
        gs.fit(A, ytr)
        model = gs.best_estimator_
        best = str(gs.best_params_)
    else:
        model = et(params).fit(A, ytr)
        best = str(params or {})
    return eval_pred(yte, model.predict(B), y_cols), best


def main():
    C.ensure_dirs()
    all50 = list(pd.read_csv(C.SENS_FILE, nrows=0).columns[3:])
    rows = []
    specs = [
        ("46dim_current", C.MODEL_TARGETS, None, "median", False),
        ("50dim_current", all50, None, "median", False),
        ("50dim_mean", all50, None, "mean", False),
        ("50dim_paper_grid", all50, None, "median", True),
    ]
    for name, cols, params, strategy, tune in specs:
        m, best = run(cols, params, strategy, tune)
        rows.append({"setup": name, **{k: round(v, 4) for k, v in m.items()},
                     "best_params": best})
        print(name, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "paper_methods_test.csv", index=False)


if __name__ == "__main__":
    main()
