import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import q3_config as C
import gen_alt_utils as G

ctx = G.load_ctx()
# 真实性检验参考集：全部 250 款真实样本
import dataio
_b = dataio.load_all()
_cols = _b.feature_cols("V3_absolute")
_prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
_im, _qt = _prep["imputer"], _prep["quantile"]
_comps = np.load(C.OUT_DIR / "pca_model.npz")["components"]
Zall = _qt.transform(_im.transform(_b.chem[_cols])) @ _comps.T
style_all = _b.style.astype(str).set_axis(_b.chem.index)
# 用主评估表（metrics_by_style.csv）；缺失时回退到旧的 hybrid 表
_main = C.OUT_DIR / "metrics_by_style.csv"
_src = _main if _main.exists() else C.OUT_DIR / "hybrid_metrics_by_style.csv"
obs = pd.read_csv(_src, encoding="utf-8-sig")
obs = obs[obs.low_confidence == False].set_index("style")
rng = np.random.default_rng(C.SEED)
rows=[]
for s in obs.index:
    Z = Zall[style_all.to_numpy() == s]
    n = len(Z)
    if n < C.RARE_THRESHOLD:
        continue
    aucs=[]
    for _ in range(200):
        # 与观测同规模：两类各抽 n 条（有放回），样本量与 step6 的等样本量协议一致
        ia = rng.choice(n, n, replace=True)
        ib = rng.choice(n, n, replace=True)
        Xn = np.vstack([Z[ia], Z[ib]])
        yn = np.r_[np.zeros(n), np.ones(n)]
        p=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),
                            Xn,yn,cv=StratifiedKFold(3,shuffle=True,random_state=int(rng.integers(1e9))),
                            method="predict_proba")[:,1]
        aucs.append(roc_auc_score(yn,p))
    q05,q50,q95=np.quantile(aucs,[0.05,0.5,0.95])
    o=float(obs.loc[s,"real_vs_fake_auc"])
    rows.append({"style":s,"n":n,"obs_auc":o,"null_q05":q05,"null_median":q50,"null_q95":q95,
                 "obs_outside_null":bool(o<q05 or o>q95)})
    print(f"{s:<18} n={n:<3} obs={o:.3f} null[{q05:.3f},{q95:.3f}] outside={o<q05 or o>q95}")
d=pd.DataFrame(rows); d.to_csv(C.OUT_DIR/"c2st_null.csv",index=False,encoding="utf-8-sig")
print("\noutside null:", int(d.obs_outside_null.sum()), "/", len(d))
print("null median range:", d.null_median.min(), d.null_median.max())
print("null q05-q95 width mean:", (d.null_q95-d.null_q05).mean())
