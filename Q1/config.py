"""Q1_v2 全局契约。

所有视图定义、目标口径、划分 seed、超参网格在跑实验前固定于此；
实验跑完后不得增删候选（防事后选优）。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "附件" / "清洗后"
OUT_DIR = ROOT / "Q1" / "results"
FIG_DIR = ROOT / "Q1" / "figures"

# 化学值取原始 1.csv：清洗表同样保留 NaN 与掩码（未做全样本事前填补），
# 这里直接读原始表以与数据契约一致；插补只在训练折内做。
CHEM_FILE = ROOT / "附件" / "1.csv"
SENS_FILE = DATA_DIR / "4_感官分析结果.csv"
COMMENT_FILE = DATA_DIR / "7_论坛评论数据.csv"
SCALE_FILE = DATA_DIR / "化学字段单位与尺度表.csv"
MAPPING_FILE = DATA_DIR / "评论指标映射.csv"

ID_COL = "beer"
STRATIFY_COL = "tasting_category_fine"

# ---------------------------------------------------------------- 目标口径
CORE_A = [
    "Malt aroma - overall", "Malt aroma - grain", "Malt aroma - bread",
    "Malt aroma - caramel", "Malt aroma - smoked",
    "Hops aroma - overall", "Hops aroma - citrus", "Hops aroma - tropical fruit",
    "Hops aroma - grassy", "Hops aroma - woody",
    "Esters aroma - overall", "Esters aroma - nail polish", "Esters aroma - banana",
    "Esters aroma - floral", "Esters aroma - fruity",
]
CORE_F = [c.replace(" aroma ", " taste ") for c in CORE_A]
CORE_TARGETS = CORE_A + CORE_F                      # 30 维，论文 Table 1 口径

# 3 个关键异味 + t2n：走绝对浓度 + 阈值判定，不进入感官预测层
THRESHOLD_TARGETS = ["Dimethyl sulfide", "Diacetyl", "4-vinyl guaiacol", "Trans-2-nonenal"]
AUX_ALL = [
    "Acidity", "Bitterness", "Sweetness", "4-vinyl guaiacol", "Diacetyl",
    "Dimethyl sulfide", "Metallic", "Stale hops", "Trans-2-nonenal", "Orange",
    "Coriander", "Clove", "Lactic acid", "Acetic acid", "Barnyard",
    "Alcohol", "Aftertaste", "Body", "Carbonation",
    "Overall appreciation (Trained panel)",
]
AUX_MODEL_TARGETS = [c for c in AUX_ALL if c not in THRESHOLD_TARGETS]   # 16 维
MODEL_TARGETS = CORE_TARGETS + AUX_MODEL_TARGETS                        # 46 维

# ---------------------------------------------------------------- 阈值判定层
# 化学列 -> (感官列, 阈值, 单位, 是否可判定)
THRESHOLD_JUDGEMENT = {
    "Dimethyl sulfide": ("Dimethyl sulfide", 17.25, "ug/L", True),
    "Diacetyl": ("Diacetyl", 0.29925, "mg/L", True),
    "4-vinylguaiacol": ("4-vinyl guaiacol", 0.206333, "mg/L", True),
    # t2n 在 8.xlsx 阈值表中缺 threshold_mean，无法做绝对判定
    "E-2-nonenal": ("Trans-2-nonenal", None, "mg/L", False),
}

# ---------------------------------------------------------------- 特征视图
# 由 scale 表生成；此处只放视图名与固定剔除项，列清单在 dataio 中解析
DERIVED_COLS = [
    "Total ester compounds", "Total hop aroma compounds",
    "Total acids", "Total sulfur compounds",
]
VIEWS = {
    "V1_paper_all": "全部 231 列（释放尺度，含派生加和）",
    "V2_scale_aware": "231 - 4 派生 = 227 列（主视图）",
    "V3_absolute": "仅 70 个绝对列（校准集，真实浓度）",
    "V4_relative": "仅 157 个相对列（142 未校准 log + 15 半定量）",
    "V5_restored": "227 列还原浓度（log10 列取 10^x，其余原样）",
}

# ---------------------------------------------------------------- 验证协议
PAPER_TEST_SIZE = 0.30
PAPER_SEED = 0
PAPER_SPLIT = dict(test_size=PAPER_TEST_SIZE, random_state=PAPER_SEED,
                   shuffle=True, stratify=None)   # stratify 在 dataio 中补

STABILITY_SEEDS = [20261023 + i for i in range(10)]   # 10 个划分 seed
N_OUTER_FOLDS = 3                                     # Faro=3 条，硬约束
N_INNER_FOLDS = 5                                     # 论文 GridSearchCV(cv=5)

# ---------------------------------------------------------------- 超参网格
# 只保留报告正文/附录中可复现的候选；被舍弃模型的结果冻结在 results/legacy/
GRIDS = {
    "et": {
        "param_grid": {
            "max_features": [0.2, 0.4, 0.6],
            "max_depth": [6, 10, None],
            "min_impurity_decrease": [0.0, 0.05, 0.1],
        },
        "needs_scaling": False,
    },
    "rf": {
        "param_grid": {
            "max_features": [0.2, 0.4, 0.6],
            "max_depth": [6, 10, None],
            "min_impurity_decrease": [0.0, 0.05, 0.1],
        },
        "needs_scaling": False,
    },
    "hist_gb": {
        "param_grid": {"max_leaf_nodes": [15, 31], "learning_rate": [0.05, 0.1]},
        "needs_scaling": False,
    },
    "pls": {
        "param_grid": {"n_components": [2, 3, 5, 8, 12]},
        "needs_scaling": True,
    },
    "ridge": {
        "param_grid": {"alpha": [1.0, 10.0, 100.0]},
        "needs_scaling": True,
    },
    "knn": {
        "param_grid": {"n_neighbors": [5, 10, 15]},
        "needs_scaling": True,
    },
}
BASELINES = ["mean", "style_mean"]

# 最终交付模型（步骤2 网格选出的 ET 配置）
FINAL_ET_PARAMS = dict(
    n_estimators=300, bootstrap=True, max_features=0.6,
    max_depth=None, min_impurity_decrease=0.0, random_state=0, n_jobs=30,
)

# ---------------------------------------------------------------- 指标
METRICS = ["r2", "rmse", "mae"]
R2_STRONG = 0.30
R2_MEDIUM = 0.10

# 交付
N_LEVELS = 7                                   # 相对 0-6
THRESHOLD_SCHEMES = ["quantile", "kmeans", "equal_freq"]


def ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

# 步骤3 探索性边界扩展（明确标注为探索性，不与预注册网格混报）
EXPLORATORY_GRIDS = {
    "et": {
        "max_features": [0.6, 0.8, 1.0],
        "max_depth": [10, None],
        "min_samples_leaf": [1, 2, 4],
        "min_impurity_decrease": [0.0, 0.05],
    },
    "rf": {
        "max_features": [0.6, 0.8, 1.0],
        "max_depth": [10, None],
        "min_samples_leaf": [1, 2, 4],
        "min_impurity_decrease": [0.0, 0.05],
    },
}
