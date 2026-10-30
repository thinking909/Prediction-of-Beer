"""运行弱维先验支持诊断（附录 C.2 的数据来源）。

依次执行：
  _diag_rescue.py        -> q2_rescue_diagnostic.csv
  _diag_rescue2.py       -> q2_rescue_random_control.csv
  _diag_prior_global.py  -> q2_prior_tests_global.csv

依赖：results/final/q2_top3_final.csv（step8 生成）。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["_diag_rescue.py", "_diag_rescue2.py", "_diag_prior_global.py"]


def main():
    for s in SCRIPTS:
        print(f"\n===== [prior] {s} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE, check=False)
        if r.returncode != 0:
            print(f"!! {s} failed (exit {r.returncode})", flush=True)
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
