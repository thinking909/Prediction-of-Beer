"""Q2 Step 1：共表达图 + 社区 + 中心性。

主阈值：|rho| 的 95% 分位（top 5%）；Fisher z 只作统计下限。
输出：Q2/results/graph/*、Q2/figures/q2_network.png
"""
import json

import numpy as np
import pandas as pd

import q2_config as C
import dataio
import graph as G


def main():
    C.ensure_dirs()
    out = C.OUT_DIR / "graph"
    out.mkdir(parents=True, exist_ok=True)
    b = dataio.load_all()
    m2 = pd.read_csv(C.DATA_DIR / "2_化学参数Spearman相关矩阵.csv",
                     index_col=0, encoding="utf-8-sig")

    thr = G.top_quantile_threshold(m2, C.EDGE_QUANTILE)
    fisher = float(np.tanh(1.96 / np.sqrt(len(b.beer) - 3)))
    Gp, pos, neg = G.build_graph(m2, thr)
    comms, labels = G.communities(Gp)
    nodes = G.node_table(Gp, labels)
    comms_df = G.community_table(nodes)

    pd.DataFrame(pos, columns=["source", "target", "rho"]) \
        .assign(edge_type="positive") \
        .to_csv(out / "q2_graph_edges.csv", index=False)
    pd.DataFrame(neg, columns=["source", "target", "rho"]) \
        .assign(edge_type="negative") \
        .to_csv(out / "q2_graph_neg_edges.csv", index=False)
    nodes.to_csv(out / "q2_graph_nodes.csv", index=False)
    comms_df.to_csv(out / "q2_communities.csv", index=False)

    thresholds = sorted(set([round(thr, 4), *C.EDGE_THRESHOLDS]))
    sens = G.sensitivity(m2, labels, thresholds)
    sens.to_csv(out / "q2_graph_sensitivity.csv", index=False)
    G.plot_network(Gp, labels, nodes, C.FIG_DIR / "q2_network.png")

    summary = {
        "primary_threshold_top5": thr,
        "fisher_floor": fisher,
        "n_nodes": int(Gp.number_of_nodes()),
        "n_positive_edges": int(Gp.number_of_edges()),
        "n_negative_edges": int(len(neg)),
        "n_communities": int(len(comms)),
        "community_sizes": sorted([len(c) for c in comms], reverse=True),
        "derived_nodes": C.DERIVED_COLS,
        "top_degree": nodes.head(10)[["compound", "weighted_degree", "community"]]
                      .round(4).to_dict("records"),
        "top_betweenness": nodes.sort_values("betweenness", ascending=False)
                           .head(10)[["compound", "betweenness", "community"]]
                           .round(4).to_dict("records"),
        "sensitivity": sens.round(4).to_dict("records"),
    }
    (out / "graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ["primary_threshold_top5", "fisher_floor", "n_positive_edges",
                       "n_negative_edges", "n_communities", "community_sizes"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
