"""第二轮：Mixture of Factor Analyzers / Mixture PPCA 近似。"""
import gen_alt_utils as G

ctx = G.load_ctx()
for ncomp, npc in [(3, 8), (5, 8), (3, 12)]:
    d = G.generate_mfa(ctx, n_components=ncomp, n_pcs=npc)
    print(f"MFA diag{ncomp} pc{npc} -> {d.shape}", flush=True)
