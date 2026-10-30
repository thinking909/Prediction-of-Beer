"""补充图：化学类别组级重要性热图。"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import q2_config as C


def main():
    gdir = C.OUT_DIR / "group"
    fig = C.FIG_DIR
    fig.mkdir(parents=True, exist_ok=True)

    mpath = gdir / "q2_group_dr2_matrix_class.csv"
    if not mpath.exists():
        print("WARNING: group results not found; skip fig8.")
        return
    m = pd.read_csv(mpath, index_col=0, encoding="utf-8-sig")
    order = m.mean(axis=0).sort_values(ascending=False).index
    m = m[order]
    half = (len(m) + 1) // 2
    parts = [m.iloc[:half], m.iloc[half:]]
    fig1, axes = plt.subplots(1, 2, figsize=(10, 7.5),
                              gridspec_kw={"wspace": 0.70})
    vmax = float(np.nanpercentile(np.abs(m.to_numpy()), 98))
    im = None
    for k, (ax, part) in enumerate(zip(axes, parts)):
        im = ax.imshow(part.to_numpy(), cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(part.shape[1]))
        ax.set_xticklabels(part.columns, rotation=90, ha="center", fontsize=8)
        ax.set_yticks(range(len(part)))
        ax.set_yticklabels([str(t).replace("Overall appreciation (Trained panel)",
                                           "Overall appreciation") for t in part.index],
                           fontsize=9)
        ax.set_title("Targets 1–25" if k == 0 else "Targets 26–50", fontsize=11)
    fig1.subplots_adjust(left=0.10, right=0.88, bottom=0.28, top=0.86,
                         wspace=0.70)
    cax = fig1.add_axes([0.91, 0.28, 0.015, 0.58])
    cbar = fig1.colorbar(im, cax=cax)
    cbar.set_label("ΔR²", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig1.suptitle("Group-level ΔR² (chemical classes, V3)", fontsize=13, y=0.98)
    fig1.savefig(fig / "fig8_group_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print("saved fig8_group_importance.png")


if __name__ == "__main__":
    main()
