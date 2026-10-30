import shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; OUT=HERE/"results"; PY=sys.executable
shutil.copy(OUT/"generated_profiles_hybrid.csv", OUT/"generated_profiles.csv")
shutil.copy(OUT/"generated_concentration_hybrid.csv", OUT/"generated_concentration.csv")
for s in ["step3_constraints.py","step4_sensory.py"]:
    subprocess.run([PY,s],cwd=HERE,check=True,stdout=subprocess.DEVNULL)
for f in ["sensory_style_summary.csv","fingerprint_check.csv","defect_summary.csv","ood_flags.csv"]:
    shutil.copy(OUT/f, OUT/f.replace(".csv","_hybrid.csv"))
for s in ["step2_copula.py","step3_constraints.py","step4_sensory.py","step5_transfer.py","step6_evaluate.py"]:
    subprocess.run([PY,s],cwd=HERE,check=True,stdout=subprocess.DEVNULL)
print("saved hybrid summaries and restored")
