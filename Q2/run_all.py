"""Q2 一键复现（Step 0–10 + 补充图）。

用法：
  python run_all.py                 # 全流程，约 55–60 分钟
  python run_all.py --list
  python run_all.py --stages graph,univariate,figures
  python run_all.py --skip importance,dropcolumn,group
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
    ("validate", ["step0_validate.py"]),
    ("graph", ["step1_graph.py"]),
    ("univariate", ["step2_univariate.py"]),
    ("stability", ["step3_stability.py"]),
    ("importance", ["step4_importance.py"]),
    ("consensus", ["step5_consensus.py"]),
    ("dropcolumn", ["step6_dropcolumn.py"]),
    ("style", ["step7_style.py"]),
    ("finalize", ["step8_finalize.py"]),
    ("figures", ["step9_figures.py"]),
    ("group", ["step10_group_importance.py"]),
    ("prior", ["step13_prior_support.py"]),
    ("consumer", ["step11_consumer_validation.py"]),
    ("extrafigures", ["step12_extra_figures.py"]),
]
NAMES = [s for s, _ in STAGES]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="", help="逗号分隔，默认全部")
    ap.add_argument("--skip", default="", help="逗号分隔")
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
    print("主要产物：results/、figures/")


if __name__ == "__main__":
    main()
