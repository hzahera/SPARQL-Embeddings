#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from result_aware.config import DEFAULT_CONFIG_DIR, load_experiment_config, load_registry
from result_aware.results import aggregate_results
from result_aware.utils import ensure_dir


def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def zval(entry: dict[str, Any], *path: str) -> float | None:
    cur: Any = entry.get("zero_shot", {})
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return float(cur) if finite(cur) else None


def tstat(entry: dict[str, Any], section: str, key: str) -> dict[str, float] | None:
    obj = entry.get("tuned", {}).get(section, {}).get(key)
    if isinstance(obj, dict) and finite(obj.get("mean")):
        return obj
    return None


def fmt_zero(v: float | None) -> str:
    return "--" if v is None else f"{v:.4f}"


def fmt_tuned(stat: dict[str, float] | None) -> str:
    if stat is None:
        return "--"
    return f"${stat['mean']:.4f} \\pm {stat['std']:.4f}$"


def fmt_delta(z: float | None, stat: dict[str, float] | None) -> str:
    if z is None or stat is None:
        return "--"
    return f"{stat['mean'] - z:+.4f}"


def model_label(meta: dict[str, Any]) -> str:
    name = meta["display_name"]
    cite = meta.get("citation")
    return f"{name}~\\cite{{{cite}}}" if cite else name


def alignment_table(summary: dict[str, Any], registry: dict[str, Any], model_keys: list[str]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\caption{Result awareness before and after fine-tuning on \squareds{}. $\rho$ denotes Spearman correlation between embedding cosine similarity and answer-set Jaccard on held-out query pairs. $\Delta\rho$ is the improvement over the corresponding zero-shot encoder. Fine-tuned results are reported as mean $\pm$ standard deviation over five seeds.}",
        r"\label{tab:alignment}",
        r"\begin{tabular}{@{}lccc@{\hspace{1.5em}}ccc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{DBpedia} & \multicolumn{3}{c}{Wikidata} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Encoder & Zero-shot & Tuned & $\Delta\rho$ & Zero-shot & Tuned & $\Delta\rho$ \\",
        r"\midrule",
    ]
    categories = registry.get("categories", {})
    for category, category_name in categories.items():
        members = [m for m in model_keys if registry["models"][m].get("category") == category]
        if not members:
            continue
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{{category_name}}}}}\\")
        for model in members:
            meta = registry["models"][model]
            dbe = summary.get("dbpedia", {}).get(model, {})
            wde = summary.get("wikidata", {}).get(model, {})
            dz = zval(dbe, "evaluation", "alignment", "spearman_rho")
            dt = tstat(dbe, "evaluation", "alignment.spearman_rho")
            wz = zval(wde, "evaluation", "alignment", "spearman_rho")
            wt = tstat(wde, "evaluation", "alignment.spearman_rho")
            lines.append(
                f"{model_label(meta)} & {fmt_zero(dz)} & {fmt_tuned(dt)} & {fmt_delta(dz, dt)} "
                f"& {fmt_zero(wz)} & {fmt_tuned(wt)} & {fmt_delta(wz, wt)} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def retrieval_table(
    summary: dict[str, Any],
    registry: dict[str, Any],
    model_keys: list[str],
    dataset: str,
    kind: str,
) -> str:
    title = "DBpedia" if dataset == "dbpedia" else "Wikidata"
    label_ds = dataset.lower()
    ks = [3, 5, 10, 20]
    if kind == "ranking":
        metric_names = [("mrr", "MRR"), ("hits", "Hits"), ("ndcg", "nDCG")]
        caption = (
            f"Ranking metrics for result-aware retrieval on \\textbf{{{title}}}. "
            r"MRR@$k$ and Hits@$k$ use binary relevance, while nDCG@$k$ uses Jaccard overlap as graded relevance. "
            r"A candidate query $q_j$ is considered relevant to anchor $q_i$ when $J(q_i,q_j) \geq 0.8$. "
            r"Fine-tuned results are reported as mean $\pm$ standard deviation over five training seeds."
        )
        label = f"tab:{label_ds}_results"
    else:
        metric_names = [("precision", "P"), ("recall", "R"), ("f1", "F1")]
        caption = (
            f"Binary retrieval metrics for result-aware retrieval on \\textbf{{{title}}}. "
            r"Precision@$k$, Recall@$k$, and F1@$k$ use the result-equivalence criterion $J(q_i,q_j) \geq 0.8$. "
            r"Fine-tuned results are reported as mean $\pm$ standard deviation over five training seeds."
        )
        label = f"tab:{label_ds}_binary"

    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll ccc ccc ccc ccc@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{$k=3$} & \multicolumn{3}{c}{$k=5$} & \multicolumn{3}{c}{$k=10$} & \multicolumn{3}{c}{$k=20$} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}",
        r"\textbf{Model} & \textbf{Setting} " + " ".join(
            "& " + " & ".join(rf"\textbf{{{display}}}" for _, display in metric_names) for _ in ks
        ) + r" \\",
        r"\midrule",
    ]

    categories = registry.get("categories", {})
    for category, category_name in categories.items():
        members = [m for m in model_keys if registry["models"][m].get("category") == category]
        if not members:
            continue
        lines.append(rf"\multicolumn{{14}}{{@{{}}l}}{{\emph{{{category_name}}}}}\\")
        for model in members:
            meta = registry["models"][model]
            entry = summary.get(dataset, {}).get(model, {})
            zcells: list[str] = []
            tcells: list[str] = []
            for k in ks:
                for metric, _ in metric_names:
                    key = f"{metric}@{k}"
                    zcells.append(fmt_zero(zval(entry, "evaluation", kind, key)))
                    tcells.append(fmt_tuned(tstat(entry, "evaluation", f"{kind}.{key}")))
            lines.append(f"{model_label(meta)} & Zero-shot & " + " & ".join(zcells) + r" \\")
            lines.append(r"& \squareds{}-tuned & " + " & ".join(tcells) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend([r"\end{tabular}%", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def cache_table(summary: dict[str, Any], registry: dict[str, Any], model_keys: list[str], dataset: str) -> str:
    title = "DBpedia" if dataset == "dbpedia" else "Wikidata"
    lines = [
        r"\begin{table*}[!htbp]",
        r"\centering",
        rf"\caption{{Semantic query caching on \textbf{{{title}}}. The cosine threshold is selected on validation by maximizing cache hit rate subject to the configured precision constraint, and is then frozen for test. Fine-tuned results are mean $\pm$ standard deviation over five seeds.}}",
        rf"\label{{tab:{dataset}_cache}}",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Setting} & $\tau$ & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{Hit rate} & \textbf{False-hit rate} & \textbf{Safe savings} \\",
        r"\midrule",
    ]
    categories = registry.get("categories", {})
    metrics = ["precision", "recall", "f1", "hit_rate", "false_hit_rate", "safe_execution_savings"]
    for category, category_name in categories.items():
        members = [m for m in model_keys if registry["models"][m].get("category") == category]
        if not members:
            continue
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{category_name}}}}}\\")
        for model in members:
            meta = registry["models"][model]
            entry = summary.get(dataset, {}).get(model, {})
            ztau = zval(entry, "cache", "selected_threshold")
            zcells = [fmt_zero(zval(entry, "cache", "test", m)) for m in metrics]
            ttau = tstat(entry, "cache", "selected_threshold")
            tcells = [fmt_tuned(tstat(entry, "cache", f"test.{m}")) for m in metrics]
            lines.append(f"{model_label(meta)} & Zero-shot & {fmt_zero(ztau)} & " + " & ".join(zcells) + r" \\")
            lines.append(r"& \squareds{}-tuned & " + fmt_tuned(ttau) + " & " + " & ".join(tcells) + r" \\")
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CONFIG_DIR / "experiment.yaml"))
    p.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    p.add_argument("--models", nargs="+", default=["all"])
    p.add_argument("--datasets", nargs="+", default=["dbpedia", "wikidata"])
    args = p.parse_args()

    exp = load_experiment_config(args.config)
    registry = load_registry(args.config_dir)
    model_keys = list(registry["models"]) if args.models == ["all"] else args.models
    datasets = list(exp["datasets"]) if args.datasets == ["all"] else args.datasets
    results_dir = Path(exp.get("output_dir", "results")).resolve()
    summary = aggregate_results(results_dir, model_keys, datasets)
    out = ensure_dir(results_dir / "tables")

    (out / "alignment.tex").write_text(alignment_table(summary, registry, model_keys), encoding="utf-8")
    for dataset in datasets:
        (out / f"{dataset}_ranking.tex").write_text(
            retrieval_table(summary, registry, model_keys, dataset, "ranking"), encoding="utf-8"
        )
        (out / f"{dataset}_binary.tex").write_text(
            retrieval_table(summary, registry, model_keys, dataset, "binary"), encoding="utf-8"
        )
        (out / f"{dataset}_cache.tex").write_text(
            cache_table(summary, registry, model_keys, dataset), encoding="utf-8"
        )
    print(out)


if __name__ == "__main__":
    main()
