"""Q2 全局配置（只使用附件数据，不读取外部原始标签）。"""
from pathlib import Path
import sys

Q2_DIR = Path(__file__).resolve().parent
ROOT = Q2_DIR.parent
DATA_DIR = ROOT / "附件" / "清洗后"
OUT_DIR = Q2_DIR / "results"
FIG_DIR = Q2_DIR / "figures"

# 复用 Q1 的数据契约（目标列表、视图、尺度表）
Q1_DIR = ROOT / "Q1"
sys.path.insert(0, str(Q1_DIR))
import config as q1c  # noqa: E402

CORE_TARGETS = list(q1c.CORE_TARGETS)
AUX_TARGETS = list(q1c.AUX_MODEL_TARGETS)
DEFECT_TARGETS = list(q1c.THRESHOLD_TARGETS)
MODEL_TARGETS = CORE_TARGETS + AUX_TARGETS          # 46
SENSORY_TARGETS = list(q1c.AUX_ALL) + CORE_TARGETS  # 50
DERIVED_COLS = list(q1c.DERIVED_COLS)

VIEW_MAIN = "V3_absolute"
VIEW_CHECK = "V2_scale_aware"
STRATIFY_COL = q1c.STRATIFY_COL

SEED = 20261023
N_BOOT = 200
N_PERM = 20
N_FOLDS = 5
EDGE_QUANTILE = 0.95
EDGE_THRESHOLDS = (0.4, 0.5, 0.6)
FDR_Q = 0.05
BOOT_FREQ = 0.6
SIGN_CONSISTENCY = 0.80
DROP_R2 = 0.01
DROP_RMSE_PCT = 0.02
STYLE_MIN_N = 8


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
