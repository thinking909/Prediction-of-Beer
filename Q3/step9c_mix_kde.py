"""Per-style 选择：全局 KDE 1.2 vs 调参 KDE，用训练集 AUC 选，外层测试评估。"""
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KernelDensity

import q3_config as C
import gen_alt_utils as G

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PY = sys.executable
ctx = G.load_ctx()
import joblib
prep = joblib.load(OUT / "preprocess.pkl")
imp, qt = prep["imputer"], prep["quantile"]
components = np.load(OUT / "pca_model.npz")["components"]
tuned = pd.read_csv(OUT / "kde_tuning_per_style.csv")
tuned_auc = dict(zip(tuned["style"], tuned["auc_train"]))

global_df = pd.read_csv(OUT / "generated_profiles_kde_bw1p2.csv")
tuned_df = pd.read_csv(OUT / "generated_profiles_kde_tuned.csv")
global_c = pd.read_csv(OUT / "generated_concentration_kde_bw1p2.csv")
tuned_c = pd.read_csv(OUT / "generated_concentration_kde_tuned.csv")
rng = np.random.default_rng(C.SEED)
rows, sel = [], []
for s in sorted(tuned["style"]):
    ir = ctx.style_tr.to_numpy() == s
    Zs = ctx.Ztr[ir]
    kde = KernelDensity(bandwidth=1.2, kernel="gaussian").fit(Zs)
    zs = kde.sample(50, random_state=int(rng.integers(1e9)))
    Xg = G.finish(zs, s, ctx)
    Zg = qt.transform(imp.transform(Xg[ctx.cols])) @ components.T
    Xc = np.vstack([Zs, Zg]); yc = np.r_[np.zeros(len(Zs)), np.ones(len(Zg))]
    p = cross_val_predict(LogisticRegression(max_iter=3000, class_weight="balanced"),
                          Xc, yc, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                          method="predict_proba")[:, 1]
    auc_global = roc_auc_score(yc, p)
    auc_tuned = tuned_auc[s]
    use = "global" if abs(auc_global - 0.5) <= abs(auc_tuned - 0.5) else "tuned"
    sel.append({"style": s, "auc_global_train": auc_global,
                "auc_tuned_train": auc_tuned, "use": use})
    src = global_df if use == "global" else tuned_df
    src_c = global_c if use == "global" else tuned_c
    rows.append((src[src["style"] == s], src_c[src_c["style"] == s]))
    print(sel[-1], flush=True)

allx = pd.concat([r[0] for r in rows], ignore_index=True)
allc = pd.concat([r[1] for r in rows], ignore_index=True)
allx.to_csv(OUT / "generated_profiles_kde_mix.csv", index=False, encoding="utf-8-sig")
allc.to_csv(OUT / "generated_concentration_kde_mix.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(sel).to_csv(OUT / "kde_mix_selection.csv", index=False, encoding="utf-8-sig")

shutil.copy(OUT / "generated_profiles_kde_mix.csv", OUT / "generated_profiles.csv")
shutil.copy(OUT / "generated_concentration_kde_mix.csv", OUT / "generated_concentration.csv")
for s in ["step3_constraints.py", "step4_sensory.py", "step6_evaluate.py"]:
    subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
d = pd.read_csv(OUT / "metrics_by_style.csv")
x = d[d.low_confidence == False]
row = {"variant": "kde_mix", "n_styles": len(x),
       "style_hit_mean": float(x.style_hit_rate.mean()),
       "auc_mean": float(x.real_vs_fake_auc.mean()),
       "auc_median": float(x.real_vs_fake_auc.median()),
       "auc_in_0.4_0.6": int(((x.real_vs_fake_auc >= 0.4) & (x.real_vs_fake_auc <= 0.6)).sum()),
       "mmd_mean": float(x.mmd2.mean()), "coverage_mean": float(x.coverage.mean()),
       "diversity_mean": float(x.diversity_ratio.mean()),
       "corr_dist_mean": float(x.corr_dist.mean()), "ood_mean": float(x.ood_rate.mean())}
print(row)
d.to_csv(OUT / "kde_mix_metrics_by_style.csv", index=False, encoding="utf-8-sig")
old = pd.read_csv(OUT / "alt_generators_metrics.csv")
old = old[old.variant != "kde_mix"]
pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
    OUT / "alt_generators_metrics.csv", index=False, encoding="utf-8-sig")

for s in ["step2_copula.py", "step3_constraints.py", "step4_sensory.py",
          "step5_transfer.py", "step6_evaluate.py"]:
    subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
print("restored default")
