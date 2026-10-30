"""项目总一键复现脚本。

默认依次执行：
  1. 附件数据清洗（清洗脚本 → 还原脚本 → 预处理补充脚本 → 评论映射脚本 → 异常值分布图）
  2. 问题一（Q1/run_all.py）
  3. 问题二（Q2/run_all.py）
  4. 问题三（Q3/run_all.py 主流程 + C2ST null + spiking + 图表；
     默认再跑 Hybrid 生成器并恢复 v6 交付状态）

用法：
  python run_all.py
  python run_all.py --list
  python run_all.py --skip clean,q2
  python run_all.py --stages clean,q1
  python run_all.py --no-hybrid

"""
import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
PY = sys.executable
LOG = ROOT / "reproduce.log"

# 各问/清洗阶段 -------------------------------------------------------------
CLEAN_SCRIPTS = [
    ("clean_basic", ROOT / "附件" / "清洗脚本.py"),
    ("clean_restore", ROOT / "附件" / "还原脚本.py"),
    ("clean_extra", ROOT / "附件" / "预处理补充脚本.py"),
    ("clean_mapping", ROOT / "附件" / "评论映射脚本.py"),
    ("clean_outlier_fig", ROOT / "附件" / "异常值分布图.py"),
]

Q3_MAIN = [
    ("q3_main", [ROOT / "Q3" / "run_all.py"]),
]
Q3_RESTORE_V6 = [
    ("q3_restore_v6", [ROOT / "Q3" / "step2b_generate_v6.py"]),
    ("q3_restore_constraints", [ROOT / "Q3" / "step3_constraints.py"]),
    ("q3_restore_sensory", [ROOT / "Q3" / "step4_sensory.py"]),
    ("q3_restore_transfer", [ROOT / "Q3" / "step5_transfer.py"]),
    ("q3_restore_evaluate", [ROOT / "Q3" / "step6_evaluate.py"]),
]
Q3_HYBRID = [
    ("q3_alt_generators", [ROOT / "Q3" / "step7_alt_generators.py"]),
    ("q3_hybrid", [ROOT / "Q3" / "step10_rejection_hybrid.py"]),
    ("q3_hybrid_summary", [ROOT / "Q3" / "_save_hybrid_metrics.py"]),
] + Q3_RESTORE_V6
Q3_ABLATIONS = [
    ("q3_kde_knn", [ROOT / "Q3" / "step7_alt_generators.py"]),
    ("q3_mfa", [ROOT / "Q3" / "step7b_mfa.py"]),
    ("q3_cvae", [ROOT / "Q3" / "step7c_cvae.py"]),
    ("q3_alt_eval", [ROOT / "Q3" / "step8_alt_eval.py"]),
    ("q3_tune_kde", [ROOT / "Q3" / "step9_tune_kde.py"]),
    ("q3_eval_tuned", [ROOT / "Q3" / "step9b_eval_tuned.py"]),
    ("q3_mix_kde", [ROOT / "Q3" / "step9c_mix_kde.py"]),
]

REQUIRED = [
    ROOT / "附件" / "1.csv",
    ROOT / "附件" / "2.csv",
    ROOT / "附件" / "4.csv",
    ROOT / "附件" / "8.xlsx",
    ROOT / "Q1" / "run_all.py",
    ROOT / "Q2" / "run_all.py",
    ROOT / "Q3" / "run_all.py",
]

def check_env():
    missing = [pkg for pkg in
               ["numpy", "pandas", "scipy", "sklearn", "matplotlib",
                "openpyxl", "networkx", "joblib"]
               if importlib.util.find_spec(pkg) is None]
    if missing:
        raise SystemExit(f"缺少 Python 包：{', '.join(missing)}；请先 pip install -r requirements.txt")
    absent = [str(p) for p in REQUIRED if not p.exists()]
    if absent:
        raise SystemExit("缺少输入文件：\n  " + "\n  ".join(absent))


def run_one(name, cmd, log):
    print(f"\n===== [{name}] {' '.join(str(c) for c in cmd)} =====", flush=True)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    p = subprocess.Popen([PY, *[str(c) for c in cmd]], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         env=env)
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
    p.wait()
    log.flush()
    if p.returncode != 0:
        raise SystemExit(f"!! [{name}] failed (exit {p.returncode})")
    print(f"===== [{name}] done in {time.time()-t0:.1f}s =====", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stages", default="", help="逗号分隔：clean,q1,q2,q3")
    ap.add_argument("--skip", default="", help="逗号分隔")
    ap.add_argument("--no-hybrid", action="store_true", help="跳过 Q3 Hybrid 生成器")
    ap.add_argument("--with-ablations", action="store_true", help="运行 Q3 生成器消融（很慢）")
    args = ap.parse_args(argv)

    stage_names = ["clean", "q1", "q2", "q3"]
    if args.list:
        print("\n".join(stage_names))
        return
    only = [s for s in args.stages.split(",") if s]
    skip = [s for s in args.skip.split(",") if s]
    selected = [s for s in stage_names if (not only or s in only) and s not in skip]

    check_env()
    print(f"Python: {PY}")
    print(f"Project: {ROOT}")
    print("Stages: " + " -> ".join(selected))
    t_all = time.time()
    with LOG.open("w", encoding="utf-8") as log:
        log.write(f"Python: {PY}\nProject: {ROOT}\nStages: {selected}\n")
        try:
            if "clean" in selected:
                for name, script in CLEAN_SCRIPTS:
                    run_one(name, [script], log)
            if "q1" in selected:
                run_one("q1", [ROOT / "Q1" / "run_all.py"], log)
            if "q2" in selected:
                run_one("q2", [ROOT / "Q2" / "run_all.py"], log)
            if "q3" in selected:
                for name, cmd in Q3_MAIN:
                    run_one(name, cmd, log)
                if not args.no_hybrid:
                    for name, cmd in Q3_HYBRID:
                        run_one(name, cmd, log)
                if args.with_ablations:
                    for name, cmd in Q3_ABLATIONS:
                        run_one(name, cmd, log)
                    for name, cmd in Q3_RESTORE_V6:
                        run_one(name, cmd, log)
        except SystemExit as e:
            print(e)
            log.write(str(e) + "\n")
            raise
        finally:
            log.write(f"\nTotal: {(time.time()-t_all)/60:.1f} min\n")
    print(f"\nAll done in {(time.time()-t_all)/60:.1f} min")
    print(f"Log: {LOG}")
    print("主要产物：Q1/results, Q2/results, Q3/results")


if __name__ == "__main__":
    main()
