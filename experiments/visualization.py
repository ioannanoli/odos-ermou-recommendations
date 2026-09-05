"""Publication-ready plots for the controlled improvement experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.graph_visualizations import (plot_graph_backbone,
                                              plot_sku_neighborhood,
                                              strongest_edge_backbone)
from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import load_orders
from src.evaluation import chronological_order_split


OUTPUT_ROOT = Path("outputs/improvement_experiments")
PLOT_DIRECTORY = OUTPUT_ROOT / "plots"


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(ablation_results, output_path):
    metrics = {
        "Hit Rate@10": "hit_rate_at_k", "Recall@10": "recall_at_k",
        "MRR@10": "mrr_at_k", "Coverage@10": "catalog_coverage_at_k",
    }
    names = ablation_results["model"].str.replace(r"^[A-Z] ", "", regex=True)
    x = np.arange(len(names))
    width = 0.18
    fig, axis = plt.subplots(figsize=(13, 6))
    for index, (label, column) in enumerate(metrics.items()):
        axis.bar(x + (index - 1.5) * width, ablation_results[column], width, label=label)
    axis.set_xticks(x, names, rotation=22, ha="right")
    axis.set_ylabel("Score")
    axis.set_ylim(0, max(0.5, ablation_results[list(metrics.values())].max().max() * 1.15))
    axis.set_title("Development-set model comparison")
    axis.legend(ncols=2)
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_final_baseline_comparison(comparison, output_path):
    metrics = {
        "Hit Rate@10": "hit_rate_at_k", "Recall@10": "recall_at_k",
        "MRR@10": "mrr_at_k", "Coverage@10": "catalog_coverage_at_k",
    }
    labels = ["Original P2V + AA", "Selected final model"]
    x = np.arange(len(metrics))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, (_, row) in enumerate(comparison.iterrows()):
        values = [row[column] for column in metrics.values()]
        axis.bar(x + (index - 0.5) * width, values, width, label=labels[index])
    axis.set_xticks(x, metrics.keys())
    axis.set_ylabel("Test score")
    axis.set_title("Frozen final model vs original test baseline")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_segment_hit_rate(segment_results, segment_type, output_path):
    frame = segment_results[segment_results["segment_type"] == segment_type].copy()
    order = (["rare", "medium", "popular"] if segment_type == "popularity"
             else ["small", "medium", "large"])
    pivot = frame.pivot(index="model", columns="segment", values="hit_rate_at_k").fillna(0)
    pivot = pivot.reindex(columns=order, fill_value=0)
    names = pivot.index.to_series().str.replace(r"^[A-Z] ", "", regex=True)
    x = np.arange(len(pivot))
    width = 0.8 / max(len(order), 1)
    fig, axis = plt.subplots(figsize=(13, 6))
    for index, segment in enumerate(order):
        axis.bar(x + (index - (len(order) - 1) / 2) * width,
                 pivot[segment], width, label=segment.title())
    axis.set_xticks(x, names, rotation=22, ha="right")
    axis.set_ylabel("Hit Rate@10")
    axis.set_title(f"Development Hit Rate@10 by {segment_type.replace('_', ' ')}")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def plot_hyperparameter_search(search_results, output_path):
    ranked = search_results.sort_values("hit_rate_at_k", ascending=False).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(11, 5))
    colors = np.where(ranked["trial"] == 1, "#d95f02", "#1b9e77")
    axis.bar(np.arange(1, len(ranked) + 1), ranked["hit_rate_at_k"], color=colors)
    axis.set_xlabel("Configuration rank (best to worst)")
    axis.set_ylabel("Development Hit Rate@10")
    axis.set_title("Product2Vec random-search results")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output_path)


def generate_experiment_visualizations(data_path, output_root=OUTPUT_ROOT,
                                       plot_directory=PLOT_DIRECTORY, center_sku=None):
    output_root, plot_directory = Path(output_root), Path(plot_directory)
    ablations = pd.read_csv(output_root / "final" / "ablation_results.csv")
    segments = pd.read_csv(output_root / "final" / "segment_results.csv")
    search = pd.read_csv(output_root / "product2vec_search" /
                         "product2vec_search_results.csv")
    plot_model_comparison(ablations, plot_directory / "model_comparison.png")
    plot_segment_hit_rate(segments, "popularity",
                          plot_directory / "hit_rate_by_popularity.png")
    plot_segment_hit_rate(segments, "basket_size",
                          plot_directory / "hit_rate_by_cart_size.png")
    plot_hyperparameter_search(search, plot_directory / "product2vec_search.png")
    comparison_path = output_root / "final" / "baseline_comparison.csv"
    if comparison_path.exists():
        plot_final_baseline_comparison(
            pd.read_csv(comparison_path), plot_directory / "final_baseline_comparison.png"
        )

    config = json.loads((output_root / "final" / "best_dev_configuration.json")
                        .read_text(encoding="utf-8"))
    orders = load_orders(data_path)
    train, _, _ = chronological_order_split(orders)
    if config["allowed_statuses"] is not None:
        train = train[train["Order Status"].isin(config["allowed_statuses"])].copy()
    graph = CoPurchaseRecommender(
        weighting=config["graph"]["weighting"],
        min_pair_count=config["graph"]["min_pair_count"],
        half_life_months=config["graph"]["half_life_months"],
        ranking=config["graph"]["ranking"],
    ).fit(train).graph
    plot_graph_backbone(strongest_edge_backbone(graph, max_edges=100),
                        plot_directory / "selected_graph_backbone.png")
    if center_sku is None or str(center_sku) not in graph:
        center_sku = max(graph.degree, key=lambda pair: pair[1])[0]
    plot_sku_neighborhood(graph, str(center_sku),
                          plot_directory / f"selected_graph_ego_{center_sku}.png")
    return plot_directory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=Path("data/Orders-Export-2026-June-07-2054.xlsx"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--plots", type=Path, default=PLOT_DIRECTORY)
    parser.add_argument("--center-sku")
    return parser.parse_args()


def main():
    arguments = parse_args()
    generate_experiment_visualizations(arguments.data, arguments.output,
                                       arguments.plots, arguments.center_sku)


if __name__ == "__main__":
    main()
