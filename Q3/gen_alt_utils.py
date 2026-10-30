"""第一轮替代生成器：共享的数据上下文与后处理。

后处理与 step2 保持一致：
  - 低维得分 -> 70 维正态得分 -> 逐列秩变换 -> 风格经验分位数；
  - 非缺陷列方差匹配；
  - 缺陷列按风格 95% 封顶。
"""
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.mixture import BayesianGaussianMixture
from sklearn.neighbors import KernelDensity

import q3_config as C
import dataio

DEFECT_COLS = ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol", "E-2-nonenal"]


@dataclass
class Ctx:
    b: object
    cols: list
    X: pd.DataFrame
    Xtr: pd.DataFrame
    style_tr: pd.Series
    Ztr: np.ndarray
    components: np.ndarray
    mean_hat: np.ndarray
    sd_hat: np.ndarray
    marg: dict
    is_log: dict
    n_gen: dict


def load_ctx():
    b = dataio.load_all()
    cols = b.feature_cols("V3_absolute")
    X = b.chem[cols].copy()
    folds = pd.read_csv(C.OUT_DIR / "folds.csv", encoding="utf-8-sig")
    trm = folds.split == "outer_train"
    Xtr = X.loc[folds.loc[trm, "beer"].to_numpy()]
    style_tr = folds.loc[trm, "style_orig"].set_axis(Xtr.index)
    prep = joblib.load(C.OUT_DIR / "preprocess.pkl")
    imp, qt = prep["imputer"], prep["quantile"]
    components = np.load(C.OUT_DIR / "pca_model.npz")["components"]
    Ztr70 = qt.transform(imp.transform(Xtr))
    Ztr = Ztr70 @ components.T
    Ximp = pd.DataFrame(imp.transform(Xtr), columns=cols, index=Xtr.index)
    Zhat = Ztr @ components
    mean_hat = Zhat.mean(axis=0)
    sd_hat = Zhat.std(axis=0)
    sd_hat[sd_hat < 1e-8] = 1.0
    sour = [s for s in C.SOUR_FAMILY if s in style_tr.unique()]
    sour_vals = Ximp.loc[style_tr.isin(sour)] if sour else Ximp
    marg = {}
    n_gen = {}
    for s in sorted(style_tr.unique()):
        idx = style_tr[style_tr == s].index
        n = len(idx)
        if n >= C.RARE_THRESHOLD:
            marg[s] = Ximp.loc[idx]
            n_gen[s] = C.N_GEN_MAIN if s in C.MAIN_STYLES else C.N_GEN_OTHER
        else:
            marg[s] = sour_vals if s in C.SOUR_FAMILY else Ximp
            n_gen[s] = 0
    meta = pd.read_csv(C.OUT_DIR / "compound_meta.csv", encoding="utf-8-sig")
    is_log = dict(zip(meta.compound, meta.is_log.astype(bool)))
    return Ctx(b, cols, X, Xtr, style_tr, Ztr, components, mean_hat, sd_hat,
               marg, is_log, n_gen)


def finish(z, style, ctx, var_match=True, alpha=1.0):
    Z_hat = np.asarray(z) @ ctx.components
    Z_std = (Z_hat - ctx.mean_hat) / ctx.sd_hat
    n = len(Z_std)
    U = np.column_stack([rankdata(Z_std[:, j]) / (n + 1)
                         for j in range(Z_std.shape[1])])
    marg = ctx.marg[style]
    Xg = np.zeros_like(U)
    for j, c in enumerate(ctx.cols):
        v = marg[c].dropna().to_numpy(float)
        Xg[:, j] = np.quantile(v, U[:, j], method="linear") if len(v) else np.nan
    Xg = pd.DataFrame(Xg, columns=ctx.cols)
    # winsorize
    n_real = len(ctx.marg[style])
    lo_q = 0.05 if n_real < 15 else 0.025
    hi_q = 0.95 if n_real < 15 else 0.975
    for c in ctx.cols:
        v = marg[c].dropna()
        Xg[c] = Xg[c].clip(v.quantile(lo_q), v.quantile(hi_q))
    if var_match:
        for c in ctx.cols:
            if c in DEFECT_COLS:
                continue
            mu = float(marg[c].mean())
            sd = float(marg[c].std())
            sdg = float(Xg[c].std())
            if sd > 0 and sdg > 1e-12:
                Xg[c] = mu + (Xg[c] - mu) * (sd / sdg) * alpha
                Xg[c] = Xg[c].clip(float(marg[c].min()), float(marg[c].max()))
    for c in DEFECT_COLS:
        if c in ctx.cols:
            Xg[c] = np.minimum(Xg[c].to_numpy(), float(marg[c].quantile(0.95)))
    return Xg


def save_profiles(df, tag, ctx):
    out = df.copy()
    out["low_confidence"] = False
    out["sample_id"] = [
        f"{r.style}_{i:03d}" for i, r in enumerate(out[["style"]].itertuples())
    ]
    # 重新按 style 编号
    out["sample_id"] = (
        out.groupby("style").cumcount().astype(str).str.zfill(3)
        .radd(out["style"] + "_")
    )
    out = out[["style", "sample_id", "low_confidence"] + ctx.cols]
    out.to_csv(C.OUT_DIR / f"generated_profiles_{tag}.csv",
               index=False, encoding="utf-8-sig")
    conc = df.copy()
    for c in ctx.cols:
        if ctx.is_log.get(c, False):
            conc[c] = np.power(10.0, conc[c])
    conc["low_confidence"] = False
    conc["sample_id"] = (
        conc.groupby("style").cumcount().astype(str).str.zfill(3)
        .radd(conc["style"] + "_")
    )
    conc = conc[["style", "sample_id", "low_confidence"] + ctx.cols]
    conc.to_csv(C.OUT_DIR / f"generated_concentration_{tag}.csv",
                index=False, encoding="utf-8-sig")
    return out


def generate_kde(ctx, bandwidth=1.0, n_pcs=19, alpha=1.0, seed=C.SEED):
    rng = np.random.default_rng(seed)
    frames = []
    for s in sorted(ctx.marg):
        if ctx.n_gen.get(s, 0) <= 0:
            continue
        Zs = ctx.Ztr[ctx.style_tr.to_numpy() == s][:, :n_pcs]
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(Zs)
        zs = kde.sample(ctx.n_gen[s], random_state=int(rng.integers(1e9)))
        z = np.zeros((len(zs), ctx.Ztr.shape[1]))
        z[:, :n_pcs] = zs
        Xg = finish(z, s, ctx, alpha=alpha)
        Xg["style"] = s
        frames.append(Xg)
    allx = pd.concat(frames, ignore_index=True)
    if n_pcs == 19 and alpha == 1.0:
        tag = f"kde_bw{bandwidth:g}".replace(".", "p")
    else:
        tag = f"kde_bw{bandwidth:g}_pc{n_pcs}_a{alpha:g}".replace(".", "p")
    save_profiles(allx, tag, ctx)
    return allx


def generate_knn(ctx, noise=0.2, seed=C.SEED):
    rng = np.random.default_rng(seed)
    frames = []
    for s in sorted(ctx.marg):
        if ctx.n_gen.get(s, 0) <= 0:
            continue
        Zs = ctx.Ztr[ctx.style_tr.to_numpy() == s]
        idx = rng.integers(0, len(Zs), size=ctx.n_gen[s])
        z = Zs[idx] + rng.normal(0, noise, size=(ctx.n_gen[s], Zs.shape[1]))
        Xg = finish(z, s, ctx)
        Xg["style"] = s
        frames.append(Xg)
    allx = pd.concat(frames, ignore_index=True)
    save_profiles(allx, f"knn_noise{noise:g}".replace(".", "p"), ctx)
    return allx


if __name__ == "__main__":
    ctx = load_ctx()
    print("styles with generation:", [s for s, n in ctx.n_gen.items() if n > 0])

def generate_mfa(ctx, n_components=3, n_pcs=8, seed=C.SEED):
    """每风格在低维 PCA 空间拟合 Bayesian GMM（mixture PPCA 近似）。"""
    rng = np.random.default_rng(seed)
    frames = []
    tag = f"mfa_diag{n_components}_pc{n_pcs}"
    for s in sorted(ctx.marg):
        if ctx.n_gen.get(s, 0) <= 0:
            continue
        Zs = ctx.Ztr[ctx.style_tr.to_numpy() == s][:, :n_pcs]
        gmm = BayesianGaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=0.1,
            max_iter=500,
            n_init=3,
            random_state=int(rng.integers(1e9)),
        )
        gmm.fit(Zs)
        zs, _ = gmm.sample(ctx.n_gen[s])
        z = np.zeros((len(zs), ctx.Ztr.shape[1]))
        z[:, :n_pcs] = zs
        Xg = finish(z, s, ctx)
        Xg["style"] = s
        frames.append(Xg)
    allx = pd.concat(frames, ignore_index=True)
    save_profiles(allx, tag, ctx)
    return allx

def generate_cvae(ctx, beta=1.0, latent=8, epochs=800, seed=C.SEED):
    """小型 Conditional VAE：PCA 19 维 + 风格 one-hot。"""
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    torch.manual_seed(seed)
    styles = [s for s in sorted(ctx.marg) if ctx.n_gen.get(s, 0) > 0]
    style2i = {s: i for i, s in enumerate(styles)}
    mask = ctx.style_tr.isin(styles).to_numpy()
    Z = ctx.Ztr[mask]
    y = ctx.style_tr[mask].map(style2i).to_numpy()
    scaler = StandardScaler().fit(Z)
    Xs = scaler.transform(Z).astype("float32")
    S = np.eye(len(styles), dtype="float32")[y]
    Xt = torch.tensor(Xs)
    St = torch.tensor(S)
    n, d = Xt.shape

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(d + len(styles), 128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.ReLU())
            self.mu = nn.Linear(64, latent)
            self.logvar = nn.Linear(64, latent)
            self.dec = nn.Sequential(
                nn.Linear(latent + len(styles), 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))

        def encode(self, x, s):
            h = self.enc(torch.cat([x, s], dim=1))
            return self.mu(h), self.logvar(h)

        def decode(self, z, s):
            return self.dec(torch.cat([z, s], dim=1))

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i + 32]
            x, s = Xt[idx], St[idx]
            mu, logvar = model.encode(x, s)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            xhat = model.decode(z, s)
            recon = ((xhat - x) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar))
            loss = recon + beta * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    frames = []
    with torch.no_grad():
        for s in styles:
            n_gen = ctx.n_gen[s]
            z = torch.randn(n_gen, latent)
            ss = torch.tensor(np.tile(np.eye(len(styles), dtype="float32")[style2i[s]],
                                      (n_gen, 1)))
            xhat = model.decode(z, ss).numpy()
            z_scores = scaler.inverse_transform(xhat)
            Xg = finish(z_scores, s, ctx)
            Xg["style"] = s
            frames.append(Xg)
    allx = pd.concat(frames, ignore_index=True)
    tag = f"cvae_b{beta:g}".replace(".", "p")
    save_profiles(allx, tag, ctx)
    return allx

