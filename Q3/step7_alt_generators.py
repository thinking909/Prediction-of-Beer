"""第一轮替代生成器：per-style KDE 与 kNN bootstrap。"""
import gen_alt_utils as G

ctx = G.load_ctx()
print("styles:", [s for s, n in ctx.n_gen.items() if n > 0], flush=True)
for bw in [0.8, 1.2]:
    d = G.generate_kde(ctx, bandwidth=bw)
    print(f"KDE bw={bw} -> {d.shape}", flush=True)
for noise in [0.2, 0.4]:
    d = G.generate_knn(ctx, noise=noise)
    print(f"kNN noise={noise} -> {d.shape}", flush=True)
