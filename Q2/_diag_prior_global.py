"""先验检验的全局 BH 校正（跨 23 个目标的所有先验对），避免逐目标校正高估。"""
import sys
sys.path.insert(0, '../Q1')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import dataio
import q2_config as C
from _diag_rescue import PRIORS, bh_fdr

b = dataio.load_all()
X3, X2 = b.X(C.VIEW_MAIN), b.X(C.VIEW_CHECK)


def norm(s):
    return s.lower().replace("-", "").replace(" ", "").replace("_", "")


map2 = {norm(c): c for c in X2.columns}
rows = []
for t, priors in PRIORS.items():
    y = b.sens[t].to_numpy()
    for p in priors:
        if p in X3.columns:
            x, vn = X3[p].to_numpy(), "V3"
        else:
            c2 = map2.get(norm(p))
            if c2 is None:
                continue
            x, vn = X2[c2].to_numpy(), "V2"
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            continue
        r, pv = spearmanr(x[m], y[m])
        rows.append({"target": t, "compound": p, "view": vn, "rho": r, "p": pv})

d = pd.DataFrame(rows)
d["q_global"] = bh_fdr(d.p.to_numpy())
d = d.sort_values("q_global")
d.to_csv(C.OUT_DIR / "final" / "q2_prior_tests_global.csv", index=False, encoding="utf-8-sig")
print("total prior tests:", len(d))
print("global FDR<0.05:", int((d.q_global < 0.05).sum()))
sig = d[d.q_global < 0.05]
print("targets with >=1 global-sig prior:", sig.target.nunique())
for t in sig.target.unique():
    s = sig[sig.target == t].sort_values("q_global")
    print(f"{t:<38}", " | ".join(f"{r.compound}({r.rho:+.2f})" for r in s.itertuples()))
print("\n--- best per target (global q) ---")
best = d.loc[d.groupby("target").q_global.idxmin()].sort_values("q_global")
for r in best.itertuples():
    print(f"{r.target:<38} {r.compound:<22} rho={r.rho:+.2f} p={r.p:.2g} q_global={r.q_global:.3g}")

