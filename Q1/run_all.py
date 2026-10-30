"""一键复现 Q1_v2 报告中的结果（不含已冻结的 legacy 模型）。

用法：
  python run_all.py                     # 全流程，约 105 分钟
  python run_all.py --stages paper,figures
  python run_all.py --skip tier2
  python run_all.py --list

注意：
- 最终交付模型 = 46 维主模型 + 4 维回退 + acetaldehyde 浓度代理；
  50 维模型只在 paper_methods 阶段做“论文口径复现对照”，不参与交付。
- deliver.py 依赖 stability.py 生成的 oof_et_*.csv；若只跑 deliver，需先有该文件。
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ("validate", ["validate_data.py"]),
    ("paper", ["paper_protocol.py"]),
    ("paper_methods", ["paper_methods_test.py"]),
    ("tier2", ["tier2.py", "et", "rf"]),
    ("ablation", ["scale_ablation.py"]),
    ("controls", ["ablation_controls.py"]),
    ("linear_clip", ["linear_clip.py"]),
    ("stability", ["stability.py"]),
    ("paired_clip", ["paired_clip.py"]),
    ("deliver", ["deliver.py"]),
    ("fallback_4d", ["fallback_4d.py"]),
    ("acetaldehyde", ["acetaldehyde_proxy.py"]),
    ("fill_sheet", ["fill_rating_sheet.py"]),
    ("cross", ["cross_source.py"]),
    ("figures", ["make_figures.py"]),
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
    print("主要产物：results/、figures/；legacy 结果不在此脚本重建。")


if __name__ == "__main__":
    main()
