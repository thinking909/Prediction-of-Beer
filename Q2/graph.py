"""Q2 共表达图工具。

主阈值 = |rho| 的 95% 分位（top 5%）；Fisher z 只作统计下限（见 step1 报告）。
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import q2_config as C


def top_quantile_threshold(m2, q=0.95):
    x = m2.to_numpy()
    off = x[np.triu_indices_from(x, 1)]
    return float(np.quantile(np.abs(off), q))


def build_graph(m2, threshold):
    G = nx.Graph()
    G.add_nodes_from(m2.index)
    pos, neg = [], []
    names = list(m2.index)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(m2.loc[a, b])
            if r >= threshold:
                G.add_edge(a, b, weight=r, rho=r)
                pos.append((a, b, r))
            elif r <= -threshold:
                neg.append((a, b, r))
    return G, pos, neg


def communities(G, seed=0, resolution=1.0):
    comms = list(nx.algorithms.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed))
    labels = {n: k for k, nodes in enumerate(comms) for n in nodes}
    return comms, labels


def node_table(G, labels):
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G)  # unweighted
    wdeg = dict(G.degree(weight="weight"))
    rows = []
    for n in G.nodes:
        rows.append({
            "compound": n,
            "community": labels[n],
            "degree_centrality": deg[n],
            "betweenness": bet[n],
            "weighted_degree": wdeg[n],
            "is_derived": n in C.DERIVED_COLS,
        })
    return pd.DataFrame(rows).sort_values("weighted_degree", ascending=False)


def community_table(nodes):
    rows = []
    for cid, d in nodes.groupby("community"):
        top = d.sort_values("weighted_degree", ascending=False)["compound"].head(5).tolist()
        rows.append({
            "community": int(cid),
            "size": int(len(d)),
            "derived_n": int(d.is_derived.sum()),
            "mean_weighted_degree": float(d.weighted_degree.mean()),
            "top_members": " | ".join(top),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def sensitivity(m2, base_labels, thresholds):
    rows = []
    for thr in thresholds:
        G, _, _ = build_graph(m2, thr)
        _, labels = communities(G)
        common = [n for n in base_labels if n in labels]
        a = [base_labels[n] for n in common]
        b = [labels[n] for n in common]
        rows.append({
            "threshold": float(thr),
            "n_edges": int(G.number_of_edges()),
            "n_communities": int(len(set(labels.values()))),
            "nmi_vs_primary": float(normalized_mutual_info_score(a, b)),
            "ari_vs_primary": float(adjusted_rand_score(a, b)),
        })
    return pd.DataFrame(rows)


def plot_network(G, labels, nodes, out_path, top_label=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    pos = nx.spring_layout(G, seed=0, k=0.35, iterations=100)
    communities = sorted(set(labels.values()))
    cmap = plt.get_cmap("tab20")
    color = {c: cmap(i % 20) for i, c in enumerate(communities)}
    colors = [color[labels[n]] for n in G.nodes]
    sizes = [20 + 8 * nodes.set_index("compound").loc[n, "weighted_degree"] for n in G.nodes]

    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.12, width=0.5, edge_color="#777")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=sizes,
                           linewidths=0.4, edgecolors="white")
    top = nodes.sort_values("weighted_degree", ascending=False).head(top_label)
    for _, r in top.iterrows():
        ax.text(pos[r.compound][0], pos[r.compound][1], r.compound,
                fontsize=6.5, ha="center", va="center")
    ax.set_title(f"Q2 共表达图（主阈值 top 5%；{G.number_of_nodes()} 节点，"
                 f"{G.number_of_edges()} 正边，{len(communities)} 社区）")
    ax.axis("off")
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
