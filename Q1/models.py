"""模型注册表：只包含报告正文中可复现的候选与最终 ET。

被舍弃模型（GBR/XGB/AdaBoost/SVR/MLP/Ridge+交互/Lasso+交互）的代码已删除，
其结果冻结在 results/legacy/，不再由一键脚本生成。
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C

# 内层 GridSearchCV 的并行策略：树模型并行在目标维（n_jobs=30），所以外层串行
INNER_JOBS = {"et": 1, "rf": 1, "hist_gb": 1, "pls": 5, "ridge": 5, "knn": 5}


class StyleMeanRegressor(BaseEstimator, RegressorMixin):
    """零化学信息基线：训练折内同风格目标均值，未见风格回退全局均值。"""

    def fit(self, X, y, style=None):
        if style is None:
            raise ValueError("style 必须传入")
        y = np.asarray(y, dtype=float)
        self.global_mean_ = y.mean(axis=0)
        self.style_mean_ = {s: y[style == s].mean(axis=0) for s in np.unique(style)}
        return self

    def predict(self, X, style=None):
        if style is None:
            raise ValueError("style 必须传入")
        out = np.tile(self.global_mean_, (len(X), 1))
        for i, s in enumerate(style):
            if s in self.style_mean_:
                out[i] = self.style_mean_[s]
        return out


def final_et():
    """最终交付模型（步骤2 网格选出的共享配置）。"""
    return ExtraTreesRegressor(**C.FINAL_ET_PARAMS)


def final_pipeline():
    """部署管线：折外评估时请手动在训练折上拟合 imputer。"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", final_et()),
    ])


def _factory(name):
    if name == "et":
        return ExtraTreesRegressor(n_estimators=300, bootstrap=True,
                                   random_state=0, n_jobs=30)
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=30)
    if name == "hist_gb":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=300, early_stopping=True,
                                          n_iter_no_change=20, random_state=0),
            n_jobs=30)
    if name == "pls":
        return PLSRegression(scale=False)
    if name == "ridge":
        return Ridge()
    if name == "knn":
        return KNeighborsRegressor(n_jobs=1)
    raise ValueError(name)


def grid_for(name):
    prefix = "model__estimator__" if name == "hist_gb" else "model__"
    return {prefix + k: v for k, v in C.GRIDS[name]["param_grid"].items()}


def build_pipeline(name):
    """返回 (pipeline, grid, inner_jobs)。预处理全部在 pipeline 内，保证折内拟合。"""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if C.GRIDS[name]["needs_scaling"]:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", _factory(name)))
    return Pipeline(steps), grid_for(name), INNER_JOBS[name]
