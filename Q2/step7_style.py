"""Q2 Step 7：风格偏相关 + 风格内分层 + bootstrap CI。

- 风格：按原标签处理（未合并 n<8 的小风格；早期描述已不符合代码）
- 偏 Spearman：对秩做风格哑变量回归，取残差相关
- 200 次风格内 bootstrap
- 风格内 Fisher z 作稳健性对照
输出：Q2/results/style/*
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import q2_config as C
import dataio

warnings.filterwarnings("ignore")
B_BOOT = 200
EXAMPLES = ["Acidity", "Hops aroma - overall", "Esters aroma - floral", "Dimethyl sulfide"]


def partial_rho(x, y, D):
    rx, ry = rankdata(x), rankdata(y)
    bx = np.linalg.lstsq(D, rx, rcond=None)[0]
    by = np.linalg.lstsq(D, ry, rcond=None)[0]
    return float(np.corrcoef(rx - D @ bx, ry - D @ by)[0, 1])


def within_style_rho(x, y, style, min_n=8):
    zs, ws = [], []
    for s in np.unique(style):
        m = style == s
        if m.sum() < min_n:
            continue
        r = spearmanr(x[m], y[m]).statistic
        if np.isfinite(r) and abs(r) < 1:
            zs.append(np.arctanh(r))
            ws.append(m.sum() - 3)
    if not zs:
        return np.nan
    return float(np.tanh(np.average(zs, weights=ws)))


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "style"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    stage2 = pd.read_csv(C.OUT_DIR / "dropcolumn" / "q2_top3_stage2.csv")
    style = b.sens[C.STRATIFY_COL].astype(str)
    counts = style.value_counts()
    grouped = style.where(style.map(counts) >= C.STYLE_MIN_N, "Other")
    # 全哑变量、无截距：等价于按风格组均值投影；不能用 drop_first+无截距
    D = pd.get_dummies(grouped, drop_first=False).to_numpy(float)
    groups = np.array(sorted(grouped.unique()))
    rng = np.random.default_rng(C.SEED)

    rows = []
    for view, key in [("V3", C.VIEW_MAIN), ("V2", C.VIEW_CHECK)]:
        X = b.X(key)
        print(f"[{view}] start", flush=True)
        for t in C.SENSORY_TARGETS:
            d = stage2[(stage2.view == view) & (stage2.target == t)]
            y = b.sens[t].to_numpy()
            for _, r in d.iterrows():
                c = r.compound
                if c not in X.columns:
                    continue
                x = X[c].to_numpy()
                mask = np.isfinite(x) & np.isfinite(y)
                xm, ym, sm = x[mask], y[mask], style.to_numpy()[mask]
                Dm = pd.get_dummies(pd.Series(sm), drop_first=False).to_numpy(float)
                raw = spearmanr(xm, ym).statistic
                adj = partial_rho(xm, ym, Dm)
                # 风格内 bootstrap
                boots = []
                for _ in range(B_BOOT):
                    idx = []
                    for g in np.unique(sm):
                        gi = np.where(sm == g)[0]
                        idx.extend(rng.choice(gi, len(gi), replace=True))
                    idx = np.array(idx)
                    Db = pd.get_dummies(pd.Series(sm[idx]), drop_first=False).to_numpy(float)
                    if Db.shape[1] == 0 or len(np.unique(sm[idx])) < 2:
                        continue
                    boots.append(partial_rho(xm[idx], ym[idx], Db))
                lo, hi = (np.percentile(boots, [2.5, 97.5])
                          if len(boots) > 10 else (np.nan, np.nan))
                wi = within_style_rho(xm, ym, sm)
                if r.status_stage2 == "definitional":
                    flag = "definitional"
                    status3 = "definitional"
                elif abs(adj) < 0.1 and not (lo <= 0 <= hi):
                    flag = "uncertain"
                    status3 = "exploratory" if r.status_stage2 == "stable_candidate" else r.status_stage2
                elif abs(adj) < 0.1 and (lo <= 0 <= hi):
                    flag = "style_proxy"
                    status3 = "exploratory"
                elif abs(adj) >= 0.1 and not (lo <= 0 <= hi):
                    flag = "style_conditional"
                    status3 = r.status_stage2
                else:
                    flag = "uncertain"
                    status3 = "exploratory" if r.status_stage2 == "stable_candidate" else r.status_stage2
                rows.append({
                    **r.to_dict(),
                    "rho_raw": float(raw),
                    "rho_adjusted": float(adj),
                    "boot_lo": float(lo) if np.isfinite(lo) else np.nan,
                    "boot_hi": float(hi) if np.isfinite(hi) else np.nan,
                    "within_style_rho": float(wi),
                    "style_flag": flag,
                    "status_stage3": status3,
                })
        print(f"[{view}] done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "q2_style_sensitivity.csv", index=False)
    res.to_csv(out / "q2_top3_stage3.csv", index=False)

    summary = {}
    for v in ["V3", "V2"]:
        d = res[res.view == v]
        summary[v] = {
            "rows": int(len(d)),
            "style_flag": d.style_flag.value_counts().to_dict(),
            "status_stage3": d.status_stage3.value_counts().to_dict(),
            "targets_with_stable": int(d[d.status_stage3 == "stable_candidate"].target.nunique()),
            "downgraded": int(((d.status_stage2 == "stable_candidate") &
                               (d.status_stage3 == "exploratory")).sum()),
        }
    ex = {}
    for v in ["V3", "V2"]:
        for t in EXAMPLES:
            d = res[(res.view == v) & (res.target == t)]
            ex[f"{v}|{t}"] = d[["compound", "rho_raw", "rho_adjusted",
                                "boot_lo", "boot_hi", "within_style_rho",
                                "style_flag", "status_stage3"]].round(3).to_dict("records")
    summary["examples"] = ex
    (out / "step7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
