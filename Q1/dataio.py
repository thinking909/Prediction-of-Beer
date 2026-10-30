"""数据装载、视图构建与划分协议（唯一入口，保证所有实验同源同折）。

化学值来自原始 1.csv（保留 73 个缺失），插补只在训练折内做；
清洗表同样保留 NaN 并附缺失掩码，二者在非缺失单元上完全一致。
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import config as C

LOG_CATEGORIES = {"log10还原", "未校准(log10还原)"}


@dataclass
class Bundle:
    chem: pd.DataFrame            # beer 索引；释放尺度（论文口径），保留缺失
    chem_restored: pd.DataFrame   # beer 索引；还原浓度（log10 列取 10^x）
    sens: pd.DataFrame            # beer 索引；50 维感官标签（标准化值）
    comments: pd.DataFrame        # beer 索引；7.csv 评论描述符
    scale: pd.DataFrame           # 231 行：尺度类别 / 绝对相对 / 阈值
    mapping: pd.DataFrame         # 7.csv -> 4.csv 映射

    @property
    def beer(self) -> np.ndarray:
        return self.chem.index.to_numpy()

    @property
    def style(self) -> pd.Series:
        return self.sens[C.STRATIFY_COL]

    def targets(self, kind: str = "core") -> list:
        if kind == "core":
            return list(C.CORE_TARGETS)
        if kind == "aux_all":
            return list(C.AUX_ALL)
        if kind == "aux_model":
            return list(C.AUX_MODEL_TARGETS)
        if kind == "model":
            return list(C.MODEL_TARGETS)
        raise ValueError(kind)

    def feature_cols(self, view: str) -> list:
        s = self.scale
        absolute = s.loc[s["绝对/相对"] == "绝对", "列名"].tolist()
        relative = s.loc[s["绝对/相对"] == "相对", "列名"].tolist()
        derived = s.loc[s["绝对/相对"] == "派生", "列名"].tolist()
        allc = s["列名"].tolist()
        if view == "V1_paper_all":
            return allc
        if view == "V2_scale_aware":
            return [c for c in allc if c not in derived]
        if view == "V3_absolute":
            return absolute
        if view == "V4_relative":
            return relative
        if view == "V5_restored":
            return [c for c in allc if c not in derived]
        raise ValueError(view)

    def X(self, view: str) -> pd.DataFrame:
        src = self.chem_restored if view == "V5_restored" else self.chem
        return src[self.feature_cols(view)].copy()

    def Y(self, kind: str = "core") -> pd.DataFrame:
        return self.sens[self.targets(kind)].copy()

    def paper_split(self):
        """论文协议：70/30 单切，分层于 tasting_category_fine。"""
        idx = np.arange(len(self.beer))
        tr, te = train_test_split(
            idx, test_size=C.PAPER_TEST_SIZE, random_state=C.PAPER_SEED,
            shuffle=True, stratify=self.style.to_numpy(),
        )
        return tr, te

    def outer_folds(self, seed: int):
        skf = StratifiedKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=seed)
        return list(skf.split(np.zeros(len(self.beer)), self.style.to_numpy()))


def _read_csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def load_all() -> Bundle:
    scale = _read_csv(C.SCALE_FILE)
    rename = dict(zip(scale["原始名"], scale["列名"]))
    raw = _read_csv(C.CHEM_FILE).rename(columns=rename)
    chem = raw.set_index(C.ID_COL).sort_index()[scale["列名"]].copy()

    restored = chem.copy()
    log_cols = scale.loc[scale["尺度类别"].isin(LOG_CATEGORIES), "列名"].tolist()
    restored[log_cols] = np.power(10.0, chem[log_cols])

    sens = _read_csv(C.SENS_FILE).set_index(C.ID_COL).sort_index()
    sens = sens.drop(columns=[c for c in ("beer_id",) if c in sens.columns])
    comments = _read_csv(C.COMMENT_FILE).set_index(C.ID_COL).sort_index()
    comments = comments.drop(columns=[c for c in ("beer_id", C.STRATIFY_COL)
                                      if c in comments.columns])
    mapping = _read_csv(C.MAPPING_FILE)

    b = Bundle(chem=chem, chem_restored=restored, sens=sens,
               comments=comments, scale=scale, mapping=mapping)
    return _validate(b)


def _validate(b: Bundle) -> Bundle:
    assert len(b.chem) == len(b.chem_restored) == len(b.sens) == len(b.comments) == 250
    assert b.chem.index.is_unique and b.sens.index.is_unique
    assert set(b.chem.index) == set(b.sens.index) == set(b.comments.index)
    assert list(b.chem.index) == list(b.sens.index) == list(b.comments.index)
    missing = [c for c in C.MODEL_TARGETS if c not in b.sens.columns]
    assert not missing, missing
    assert set(b.chem.columns) == set(b.scale["列名"]), set(b.chem.columns) ^ set(b.scale["列名"])
    assert list(b.chem.columns) == list(b.chem_restored.columns)
    return b


def prepare(b: Bundle, view: str, target_kind: str = "core"):
    X = b.X(view)
    Y = b.Y(target_kind)
    assert list(X.index) == list(Y.index)
    return X, Y
