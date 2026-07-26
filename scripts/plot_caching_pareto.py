#!/usr/bin/env python3
"""Utility-vs-theta figure for the caching experiment (paper fig:caching).

Two panels (DBpedia, Wikidata), one line per model: utility = hit_rate *
quality, plotted across the theta sweep at the headline K. Reads the
per-KG pareto CSVs scripts/run_caching_experiment.py already writes
(model,theta,hit_rate,quality) -- run that script first for each KG at
the paper's headline --min-answer-size setting.

Usage:
    python scripts/plot_caching_pareto.py \
        --dbpedia results/caching/caching_dbpedia_pareto.csv \
        --wikidata results/caching/caching_wikidata_pareto.csv \
        --highlight SparqlCE E5-ft \
        --out results/caching/caching_pareto_utility.pdf
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pareto(path):
    """{model: {theta: utility}} from a caching_<kg>_pareto.csv."""
    series = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            hr, q = float(row["hit_rate"]), float(row["quality"])
            series[row["model"]][float(row["theta"])] = hr * q
    return series


def plot_panel(ax, series, highlight, title):
    for name, points in series.items():
        thetas = sorted(points)
        util = [points[th] for th in thetas]
        is_hl = name in highlight
        ax.plot(thetas, util,
                marker="o", ms=4,
                lw=2.6 if is_hl else 1.0,
                alpha=1.0 if is_hl else 0.55,
                zorder=5 if is_hl else 2,
                label=name)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Utility (hit rate $\\times$ quality)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbpedia", required=True, help="caching_dbpedia_pareto.csv")
    ap.add_argument("--wikidata", required=True, help="caching_wikidata_pareto.csv")
    ap.add_argument("--highlight", nargs="+", default=["SparqlCE", "E5-ft"],
                     help="model names to draw bold/opaque; all others thin/faded")
    ap.add_argument("--out", default="results/caching/caching_pareto_utility.pdf")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_panel(axes[0], load_pareto(args.dbpedia), args.highlight, "DBpedia")
    plot_panel(axes[1], load_pareto(args.wikidata), args.highlight, "Wikidata")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
