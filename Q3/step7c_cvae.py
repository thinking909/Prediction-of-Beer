"""第三轮：Conditional VAE（PCA 19 + 风格 one-hot）。"""
import gen_alt_utils as G

ctx = G.load_ctx()
for beta in [0.5, 1.0, 2.0]:
    d = G.generate_cvae(ctx, beta=beta)
    print(f"CVAE beta={beta} -> {d.shape}", flush=True)
