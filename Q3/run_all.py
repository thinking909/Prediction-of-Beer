"""Q3 一键复现（E1 准备 + E2 生成 + E3 缺陷 + E4 感官 + E5 迁移 + E6 评价）。

用法：
  python run_all.py
  python run_all.py --list
  python run_all.py --stages prepare,copula,constraints
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

STAGES = [
    ("prepare", ["step1_prepare.py"]),
    ("copula", ["step2_copula.py"]),
    ("v6", ["step2b_generate_v6.py"]),
    ("constraints", ["step3_constraints.py"]),
    ("sensory", ["step4_sensory.py"]),
    ("transfer", ["step5_transfer.py"]),
    ("evaluate", ["step6_evaluate.py"]),
    ("c2st", ["step11_c2st_null.py"]),
    ("spiking", ["step12_spiking.py"]),
    ("figures", ["step13_figures.py"]),
]
NAMES = [s for s, _ in STAGES]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="")
    ap.add_argument("--skip", default="")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        print("\n".join(NAMES))
        return
    only = [s for s in args.stages.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    selected = [(n, c) for n, c in STAGES
                if (not only or n in only) and n not in skip]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"Python: {PY}")
    print("Stages: " + " -> ".join(n for n, _ in selected))
    t_all = time.time()
    for name, cmd in selected:
        t0 = time.time()
        print(f"\n===== [{name}] {' '.join(cmd)} =====", flush=True)
        r = subprocess.run([PY, str(HERE / cmd[0]), *cmd[1:]],
                           cwd=HERE, env=env)
        if r.returncode != 0:
            print(f"!! stage {name} failed (exit {r.returncode})", flush=True)
            sys.exit(r.returncode)
        print(f"===== [{name}] done in {time.time()-t0:.1f}s =====", flush=True)
    print(f"\nAll stages done in {(time.time()-t_all)/60:.1f} min")
    print("Outputs: results/, figures/")


if __name__ == "__main__":
    main()
