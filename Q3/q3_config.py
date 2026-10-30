"""Q3 全局配置：只使用附件数据，复用 Q1 的数据契约。"""
from pathlib import Path
import sys

Q3_DIR = Path(__file__).resolve().parent
ROOT = Q3_DIR.parent
Q1_DIR = ROOT / "Q1"
Q2_DIR = ROOT / "Q2"

sys.path.insert(0, str(Q1_DIR))
import config as q1c  # noqa: E402

DATA_DIR = ROOT / "附件" / "清洗后"
OUT_DIR = Q3_DIR / "results"
FIG_DIR = Q3_DIR / "figures"
Q2_FINAL = Q2_DIR / "results" / "final"

SEED = 20261025
OUTER_TEST_SIZE = 0.20
N_INNER = 5
K_PCA_DEFAULT = 15
K_PCA_RANGE = list(range(10, 21))
N_GEN_MAIN = 100
N_GEN_OTHER = 50
N0_SHRINK = 15

# 主案例
MAIN_STYLES = ["Tripel", "Kriek", "Lambic"]
TRANSFER_CASES = [("Pils/Lager", "Tripel"), ("Blond", "Tripel"), ("Kriek", "Lambic")]

# 酸啤族：用于族级层次收缩
SOUR_FAMILY = ["Kriek", "Lambic", "Faro", "West Flanders ale",
               "Flanders old brown", "Fruitbeer"]
RARE_THRESHOLD = 8

# 缺陷：OAV 只对前三类有效；t2n 无阈值；Stale hops 无化学列
DEFECTS = {
    "Dimethyl sulfide": {"kind": "oav", "threshold": 17.25, "unit": "ug/L"},
    "Diacetyl": {"kind": "oav", "threshold": 0.29925, "unit": "mg/L"},
    "4-vinylguaiacol": {"kind": "oav_style", "threshold": 0.206333, "unit": "mg/L"},
    "Trans-2-nonenal": {"kind": "relative", "threshold": None, "unit": "mg/L"},
    "Stale hops": {"kind": "soft_sensory", "threshold": None, "unit": None},
}

KEY_COMPOUNDS = [
    "4-vinylguaiacol", "Lactic acid", "Acetic acid", "Ethanol",
    "Ethyl acetate", "Ethyl octanoate", "Ethyl decanoate", "Glycerol",
    "Iso-alpha acids", "Dimethyl sulfide", "Diacetyl", "E-2-nonenal",
]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
