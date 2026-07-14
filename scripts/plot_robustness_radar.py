#!/usr/bin/env python3
"""Generate the RQ2 per-operator robustness radar (paper appendix figure).

Values are the per-operator mean cosine similarities from the paper's
robustness table (tables/tab-RQ2.tex), produced by
notebooks/evaluation-RQ2/{DBpedia,Wikidata}/04_eval_sparql_RQ2_*.ipynb.
Regenerated 2026-07 with readable print-size fonts and the KG-specific
footnote moved to the LaTeX caption.

Usage: python3 scripts/plot_robustness_radar.py [output.pdf]
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
})

AXES = ["Var.\nRenaming", "Triple\nReorder", "Redundant\nOPTIONAL",
        "UNION\nDup.", "KG-specific$^\\dagger$", "FILTER\n(true)", "DISTINCT\nToggle"]

DATA = {
    "DBpedia": {
        "SparqlCE (Ours)": [1.000, 1.000, 0.980, 1.000, 1.000, 1.000, 1.000],
        "TF-IDF":          [0.953, 0.997, 0.957, 0.927, 0.989, 0.981, 0.975],
        "E5":              [0.988, 0.999, 0.991, 0.989, 0.998, 0.991, 0.979],
        "Qwen3":           [0.959, 0.995, 0.991, 0.972, 0.989, 0.994, 0.989],
    },
    "Wikidata": {
        "SparqlCE (Ours)": [1.000, 0.995, 0.946, 0.995, 1.000, 1.000, 0.999],
        "TF-IDF":          [0.896, 0.987, 0.946, 0.934, 1.000, 0.932, 0.948],
        "E5":              [0.982, 0.999, 0.996, 0.975, 0.999, 0.991, 0.985],
        "Qwen3":           [0.965, 0.987, 0.994, 0.940, 0.997, 0.996, 0.986],
    },
}

STYLE = {
    "SparqlCE (Ours)": dict(color="#C03A2B", lw=2.0, ls="-",  marker="o", ms=3.5, zorder=5, fill=0.12),
    "TF-IDF":          dict(color="#8A8F98", lw=1.1, ls="--", marker="s", ms=2.8, zorder=3, fill=0),
    "E5":              dict(color="#2E5C8A", lw=1.1, ls="-.", marker="^", ms=3.0, zorder=4, fill=0),
    "Qwen3":           dict(color="#3E7A4E", lw=1.1, ls=":",  marker="D", ms=2.8, zorder=3, fill=0),
}

RMIN, RMAX = 0.85, 1.0


def main(out_path="robustness_radar.pdf"):
    n = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3), subplot_kw=dict(polar=True))

    for ax, kg in zip(axs, DATA):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for name, vals in DATA[kg].items():
            st = STYLE[name]
            v = np.clip(np.array(vals + vals[:1]), RMIN, RMAX)
            a = np.concatenate([angles, angles[:1]])
            ax.plot(a, v, color=st["color"], linewidth=st["lw"], linestyle=st["ls"],
                    marker=st["marker"], markersize=st["ms"], zorder=st["zorder"],
                    label=name, clip_on=False)
            if st["fill"]:
                ax.fill(a, v, color=st["color"], alpha=st["fill"], zorder=1)
        ax.set_ylim(RMIN, RMAX)
        ax.set_yticks([0.88, 0.92, 0.96, 1.00])
        ax.set_yticklabels(["0.88", "0.92", "0.96", "1.00"], fontsize=6.5, color="#57606A")
        ax.set_rlabel_position(12)
        ax.set_xticks(angles)
        ax.set_xticklabels(AXES, fontsize=7.5)
        ax.tick_params(pad=3)
        ax.grid(color="#D7DBE0", linewidth=0.5)
        ax.spines["polar"].set_color("#B9BFC8")
        ax.set_title(kg, fontsize=10, fontweight="bold", pad=16)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
