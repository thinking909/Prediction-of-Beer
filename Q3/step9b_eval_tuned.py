"""评估 per-style 调参后的 KDE（kde_tuned），并恢复默认输出。"""
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PY = sys.executable

shutil.copy(OUT / "generated_profiles_kde_tuned.csv", OUT / "generated_profiles.csv")
shutil.copy(OUT / "generated_concentration_kde_tuned.csv", OUT / "generated_concentration.csv")
for s in ["step3_constraints.py", "step4_sensory.py", "step6_evaluate.py"]:
    subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)

d = pd.read_csv(OUT / "metrics_by_style.csv")
d.to_csv(OUT / "kde_tuned_metrics_by_style.csv", index=False, encoding="utf-8-sig")
x = d[d.low_confidence == False]
row = {
    "variant": "kde_tuned",
    "n_styles": len(x),
    "style_hit_mean": float(x.style_hit_rate.mean()),
    "auc_mean": float(x.real_vs_fake_auc.mean()),
    "auc_median": float(x.real_vs_fake_auc.median()),
    "auc_in_0.4_0.6": int(((x.real_vs_fake_auc >= 0.4) & (x.real_vs_fake_auc <= 0.6)).sum()),
    "mmd_mean": float(x.mmd2.mean()),
    "coverage_mean": float(x.coverage.mean()),
    "diversity_mean": float(x.diversity_ratio.mean()),
    "corr_dist_mean": float(x.corr_dist.mean()),
    "ood_mean": float(x.ood_rate.mean()),
}
print(row)
old = pd.read_csv(OUT / "alt_generators_metrics.csv")
old = old[old.variant != "kde_tuned"]
pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(
    OUT / "alt_generators_metrics.csv", index=False, encoding="utf-8-sig")

for s in ["step2_copula.py", "step3_constraints.py", "step4_sensory.py",
          "step5_transfer.py", "step6_evaluate.py"]:
    subprocess.run([PY, s], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
print("restored default")
