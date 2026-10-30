"""诊断：对当前无 stable 的目标，估计两种补救能救回多少。
1) 共线组 drop-column：删除 Top-3 及其 |rho|>=0.7 的近邻后看预测贡献
2) 文献先验假设检验：每目标只测 5-7 个已知风味化合物
输出 results/final/q2_rescue_diagnostic.csv
"""
import sys
sys.path.insert(0, '../Q1')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import config as q1c
import dataio
import q2_config as C

PRIORS = {
    "Clove": ["4-vinylguaiacol", "Eugenol", "Alpha-humulene", "Caryophyllene", "Linalool"],
    "Barnyard": ["4-vinylguaiacol", "4-ethylphenol", "4-ethylguaiacol", "Acetic acid", "Lactic acid"],
    "Stale hops": ["E-2-nonenal", "3-methylbutanal", "Furfural", "5-methylfurfural", "Dimethyl sulfide"],
    "Coriander": ["Linalool", "Geraniol", "Citronellol", "Alpha-terpineol", "Eugenol"],
    "Esters aroma - floral": ["Linalool", "Geraniol", "Phenethyl acetate", "Phenethyl alcohol",
                              "Alpha-terpineol", "Benzyl alcohol", "E-Beta-damascenone"],
    "Esters taste - floral": ["Linalool", "Geraniol", "Phenethyl acetate", "Phenethyl alcohol",
                              "Alpha-terpineol", "Benzyl alcohol"],
    "Esters aroma - fruity": ["Isoamyl acetate", "Ethyl acetate", "Ethyl butyrate", "Ethyl hexanoate",
                              "Ethyl octanoate", "Ethyl decanoate", "Ethyl propionate"],
    "Esters taste - fruity": ["Isoamyl acetate", "Ethyl acetate", "Ethyl butyrate", "Ethyl hexanoate",
                              "Ethyl octanoate", "Ethyl decanoate"],
    "Esters aroma - banana": ["Isoamyl acetate", "Isoamyl alcohol", "Ethyl butyrate", "Ethyl hexanoate",
                              "Amyl acetate"],
    "Esters taste - banana": ["Isoamyl acetate", "Isoamyl alcohol", "Ethyl butyrate", "Ethyl hexanoate",
                              "Amyl acetate"],
    "Esters aroma - nail polish": ["Ethyl acetate", "Acetaldehyde", "Isoamyl acetate",
                                   "Isobutyl acetate", "Isoamyl alcohol"],
    "Malt aroma - grain": ["3-methylbutanal", "Furfural", "5-methylfurfural", "Dimethyl sulfide"],
    "Malt taste - grain": ["3-methylbutanal", "Furfural", "5-methylfurfural", "Dimethyl sulfide", "Diacetyl"],
    "Hops aroma - tropical fruit": ["Linalool", "Geraniol", "Citronellol", "Alpha-terpineol",
                                    "Isoamyl acetate", "Ethyl butyrate", "Beta-phellandrene"],
    "Hops taste - tropical fruit": ["Linalool", "Geraniol", "Citronellol", "Alpha-terpineol",
                                    "Isoamyl acetate", "Ethyl butyrate", "Beta-phellandrene"],
    "Hops taste - citrus": ["Geraniol", "Linalool", "Citronellol", "Alpha-terpineol",
                            "Beta-phellandrene", "E-Beta-ocimene"],
    "Hops taste - overall": ["Iso-alpha acids", "Alpha-humulene", "Caryophyllene", "Linalool",
                             "Geraniol", "Ethyl octanoate"],
    "Aftertaste": ["Ethanol", "Glycerol", "Ethyl octanoate", "Ethyl hexanoate", "Iso-alpha acids",
                   "Acetic acid"],
    "Overall appreciation (Trained panel)": ["Ethanol", "Glycerol", "Iso-alpha acids", "Ethyl acetate",
                                             "Ethyl octanoate", "Acetic acid", "Lactic acid",
                                             "Caloric value"],
}
# 仅在 V2 中存在的先验名
V2_ALIASES = {
    "4-ethylphenol": "4-ethylphenol",
    "4-ethylguaiacol": "4-ethylguaiacol",
    "Beta-phellandrene": "Beta-phellandrene",
}


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], 1):
        val = min(prev, p[idx] * n / (n - rank + 1))
        q[idx] = val
        prev = val
    return q


def evaluate(X, y, drop_cols, folds, n_jobs=4):
    cols = [c for c in X.columns if c not in drop_cols]
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        Xtr, Xte = X.iloc[tr][cols], X.iloc[te][cols]
        imp = SimpleImputer(strategy="median").fit(Xtr)
        et = ExtraTreesRegressor(**{**q1c.FINAL_ET_PARAMS, "n_jobs": n_jobs})
        et.fit(imp.transform(Xtr), y[tr])
        oof[te] = et.predict(imp.transform(Xte))
    return r2_score(y, oof), float(np.sqrt(mean_squared_error(y, oof)))


def main():
    b = dataio.load_all()
    X3 = b.X(C.VIEW_MAIN)
    X2 = b.X(C.VIEW_CHECK)
    final = pd.read_csv(C.OUT_DIR / "final" / "q2_top3_final.csv")
    v3 = final[final.view == "V3"]
    stab = pd.read_csv(C.OUT_DIR / "final" / "q2_stability_final.csv")
    s3 = stab[stab.view == "V3"].set_index("target")

    no_stable = [t for t in C.SENSORY_TARGETS if s3.loc[t, "n_stable"] == 0]
    print("no-stable targets:", len(no_stable), flush=True)

    rows = []
    for t in no_stable:
        sub = v3[v3.target == t].sort_values("rank")
        top = list(sub.compound)
        y = b.sens[t].to_numpy()
        ti = C.SENSORY_TARGETS.index(t)
        folds = list(KFold(3, shuffle=True, random_state=C.SEED + ti).split(X3))
        base_r2, base_rmse = evaluate(X3, y, [], folds)

        # 共线组：Top-3 + 与任一 Top-3 |rho|>=0.7 的 V3 特征
        group = set(top)
        for c in top:
            x = X3[c].to_numpy()
            for c2 in X3.columns:
                if c2 in group:
                    continue
                x2 = X3[c2].to_numpy()
                m = np.isfinite(x) & np.isfinite(x2)
                if m.sum() >= 10:
                    r = spearmanr(x[m], x2[m]).statistic
                    if abs(r) >= 0.7:
                        group.add(c2)
        grp_r2, grp_rmse = evaluate(X3, y, list(group), folds)
        d_r2 = base_r2 - grp_r2
        d_rmse = (grp_rmse - base_rmse) / max(base_rmse, 1e-9)

        # 先验检验（V3 优先，缺失时用 V2），双侧 Spearman + 目标内 BH
        prior_rows = []
        for p in PRIORS.get(t, []):
            for X, vn in ((X3, "V3"), (X2, "V2")):
                if p in X.columns:
                    x = X[p].to_numpy()
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() >= 10:
                        r, pv = spearmanr(x[m], y[m])
                        prior_rows.append((p, vn, r, pv))
                    break
            else:
                # 尝试别名匹配（大小写/连字符不同）
                hit = None
                for c2 in X2.columns:
                    key = c2.lower().replace("-", "").replace(" ", "")
                    if key == p.lower().replace("-", "").replace(" ", ""):
                        hit = c2
                        break
                if hit:
                    x = X2[hit].to_numpy()
                    m = np.isfinite(x) & np.isfinite(y)
                    if m.sum() >= 10:
                        r, pv = spearmanr(x[m], y[m])
                        prior_rows.append((p, "V2", r, pv))
        if prior_rows:
            qs = bh_fdr([r[3] for r in prior_rows])
            best = min(range(len(prior_rows)), key=lambda i: qs[i])
            best_prior = prior_rows[best][0]
            best_rho = prior_rows[best][2]
            best_q = float(qs[best])
            n_prior_hit = int(sum(q < 0.05 for q in qs))
        else:
            best_prior, best_rho, best_q, n_prior_hit = None, np.nan, np.nan, 0

        rows.append({
            "target": t,
            "current_n_stable": 0,
            "current_max_adj_rho": float(sub.rho_adjusted.abs().max()),
            "base_r2": base_r2,
            "group_size": len(group),
            "group_drop_dr2": d_r2,
            "group_drop_drmse_pct": d_rmse,
            "group_drop_pass": bool(d_r2 > C.DROP_R2 or d_rmse > C.DROP_RMSE_PCT),
            "prior_best": best_prior,
            "prior_rho": best_rho,
            "prior_q": best_q,
            "prior_hit": bool(best_q < 0.05) if np.isfinite(best_q) else False,
            "n_prior_hit": n_prior_hit,
        })
        print(f"{t:<38} dR2={d_r2:+.4f} dRMSE={d_rmse*100:+.2f}% "
              f"grp={len(group):<2} prior={best_prior} rho={best_rho:+.2f} q={best_q:.3g}",
              flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(C.OUT_DIR / "final" / "q2_rescue_diagnostic.csv", index=False, encoding="utf-8-sig")
    print("\n== summary ==")
    print("group-drop pass:", int(out.group_drop_pass.sum()), "/", len(out))
    print("prior hit:", int(out.prior_hit.sum()), "/", len(out))
    print("either:", int((out.group_drop_pass | out.prior_hit).sum()), "/", len(out))
    print("both:", int((out.group_drop_pass & out.prior_hit).sum()), "/", len(out))


if __name__ == "__main__":
    main()

