"""第一轮替代生成器评测：KDE / kNN vs 当前 Copula（方差匹配）。"""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PY = sys.executable
tags = ["kde_bw0p8", "kde_bw1p2", "knn_noise0p2", "knn_noise0p4",
        "mfa_diag3_pc8", "mfa_diag5_pc8", "mfa_diag3_pc12",
        "cvae_b0p5", "cvae_b1", "cvae_b2"]


def summarize(df):
    d = df[df.low_confidence == False].copy()
    return {
        "n_styles": len(d),
        "style_hit_mean": float(d.style_hit_rate.mean()),
        "auc_mean": float(d.real_vs_fake_auc.mean()),
        "auc_median": float(d.real_vs_fake_auc.median()),
        "auc_in_0.4_0.6": int(((d.real_vs_fake_auc >= 0.4) & (d.real_vs_fake_auc <= 0.6)).sum()),
        "mmd_mean": float(d.mmd2.mean()),
        "coverage_mean": float(d.coverage.mean()),
        "diversity_mean": float(d.diversity_ratio.mean()),
        "corr_dist_mean": float(d.corr_dist.mean()),
        "ood_mean": float(d.ood_rate.mean()),
    }


rows = []
# 当前 copula 结果
cur = pd.read_csv(OUT / "metrics_by_style.csv")
rows.append({"variant": "copula_var", **summarize(cur)})
print("copula_var", rows[-1], flush=True)

for tag in tags:
    print("\n====", tag, flush=True)
    shutil.copy(OUT / f"generated_profiles_{tag}.csv", OUT / "generated_profiles.csv")
    shutil.copy(OUT / f"generated_concentration_{tag}.csv", OUT / "generated_concentration.csv")
    for s in ["step3_constraints.py", "step4_sensory.py", "step6_evaluate.py"]:
        subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    d = pd.read_csv(OUT / "metrics_by_style.csv")
    row = {"variant": tag, **summarize(d)}
    rows.append(row)
    print(tag, row, flush=True)

pd.DataFrame(rows).to_csv(OUT / "alt_generators_metrics.csv", index=False, encoding="utf-8-sig")
print("\n", pd.DataFrame(rows).to_string(index=False))

# 恢复默认 copula 输出（不递归调用 run_all）
for s in ["step2_copula.py", "step3_constraints.py", "step4_sensory.py",
          "step5_transfer.py", "step6_evaluate.py"]:
    subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
print("restored default outputs")
