"""Per-style KDE 调参：bandwidth × n_pcs × variance alpha。"""
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KernelDensity

import q3_config as C
import gen_alt_utils as G

ctx = G.load_ctx()
prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
imp, qt = prep["imputer"], prep["quantile"]
components = np.load(C.OUT_DIR / "pca_model.npz")["components"]

grid = list(itertools.product(
    [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    [8, 12, 19],
    [0.8, 1.0, 1.2],
))
rows = []
frames = []
rng = np.random.default_rng(C.SEED)
for s in sorted(ctx.marg):
    if ctx.n_gen.get(s, 0) <= 0:
        continue
    ir = ctx.style_tr.to_numpy() == s
    Zs = ctx.Ztr[ir]
    best = None
    for bw, npc, alpha in grid:
        kde = KernelDensity(bandwidth=bw, kernel="gaussian").fit(Zs[:, :npc])
        zs = kde.sample(50, random_state=int(rng.integers(1e9)))
        z = np.zeros((len(zs), ctx.Ztr.shape[1]))
        z[:, :npc] = zs
        Xg = G.finish(z, s, ctx, alpha=alpha)
        Zg = qt.transform(imp.transform(Xg[ctx.cols])) @ components.T
        Xc = np.vstack([Zs, Zg])
        yc = np.r_[np.zeros(len(Zs)), np.ones(len(Zg))]
        cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        p = cross_val_predict(clf, Xc, yc, cv=cv, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(yc, p))
        score = abs(auc - 0.5)
        if best is None or score < best["score"]:
            best = {"style": s, "bandwidth": bw, "n_pcs": npc,
                    "alpha": alpha, "auc_train": auc, "score": score}
    rows.append(best)
    # 用最优参数生成最终样本
    Zs_best = Zs[:, :best["n_pcs"]]
    kde = KernelDensity(bandwidth=best["bandwidth"], kernel="gaussian").fit(Zs_best)
    zs = kde.sample(ctx.n_gen[s], random_state=int(rng.integers(1e9)))
    z = np.zeros((len(zs), ctx.Ztr.shape[1]))
    z[:, :best["n_pcs"]] = zs
    Xg = G.finish(z, s, ctx, alpha=best["alpha"])
    Xg["style"] = s
    frames.append(Xg)
    print(s, best, flush=True)

pd.DataFrame(rows).to_csv(C.OUT_DIR / "kde_tuning_per_style.csv",
                          index=False, encoding="utf-8-sig")
allx = pd.concat(frames, ignore_index=True)
G.save_profiles(allx, "kde_tuned", ctx)
print("saved kde_tuned", allx.shape)
