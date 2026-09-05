"""Generate readable visual summaries of the product co-purchase graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.copurchase_recommender import build_copurchase_graph
from src.data_loader import load_orders


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs/visualizations")


def graph_statistics(graph):
    """Return core size, sparsity, component, and degree statistics."""
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    component_sizes = [len(component) for component in nx.connected_components(graph)]
    rows = [
        ("nodes", graph.number_of_nodes()),
        ("edges", graph.number_of_edges()),
        ("density", nx.density(graph)),
        ("connected_components", len(component_sizes)),
        ("isolated_nodes", nx.number_of_isolates(graph)),
        ("largest_component_nodes", max(component_sizes, default=0)),
        ("mean_degree", degrees.mean() if degrees.size else 0.0),
        ("median_degree", np.median(degrees) if degrees.size else 0.0),
        ("maximum_degree", degrees.max() if degrees.size else 0.0),
    ]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def strongest_edge_backbone(graph, max_edges=100):
    """Select the strongest observed relationships for a readable global view."""
    edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (-edge[2].get("count", 0), -edge[2].get("weight", 0.0),
                          str(edge[0]), str(edge[1])),
    )[:max_edges]
    backbone = nx.Graph()
    backbone.add_edges_from(edges)
    for node in backbone:
        backbone.nodes[node].update(graph.nodes[node])
    return backbone


def neighborhood_subgraph(graph, center_sku, direct_limit=10, second_hop_limit=2):
    """Select a bounded weighted two-hop neighborhood and node-level labels."""
    center_sku = str(center_sku)
    if center_sku not in graph:
        raise ValueError(f"SKU {center_sku!r} is not present in the graph.")
    direct = sorted(
        graph[center_sku],
        key=lambda node: (-graph[center_sku][node].get("weight", 0.0), str(node)),
    )[:direct_limit]
    levels = {center_sku: 0, **{node: 1 for node in direct}}
    selected = {center_sku, *direct}
    for neighbor in direct:
        second_hop = sorted(
            (node for node in graph[neighbor] if node not in selected),
            key=lambda node: (-graph[neighbor][node].get("weight", 0.0), str(node)),
        )[:second_hop_limit]
        for node in second_hop:
            selected.add(node)
            levels.setdefault(node, 2)
    return graph.subgraph(selected).copy(), levels


def _edge_widths(graph, minimum=0.5, maximum=5.0):
    weights = np.asarray([data.get("weight", 0.0) for _, _, data in graph.edges(data=True)])
    if not weights.size or weights.max() == weights.min():
        return [minimum + 1.0] * graph.number_of_edges()
    scaled = (weights - weights.min()) / (weights.max() - weights.min())
    return (minimum + scaled * (maximum - minimum)).tolist()


def plot_graph_backbone(graph, output_path):
    """Plot the strongest global co-purchase relationships."""
    backbone = strongest_edge_backbone(graph)
    figure, axis = plt.subplots(figsize=(16, 11))
    positions = nx.spring_layout(backbone, seed=42, weight="weight", k=0.65)
    degrees = dict(backbone.degree())
    node_sizes = [80 + 35 * degrees[node] for node in backbone]
    nx.draw_networkx_edges(backbone, positions, ax=axis, alpha=0.28,
                           edge_color="#64748b", width=_edge_widths(backbone, 0.4, 3.5))
    nx.draw_networkx_nodes(backbone, positions, ax=axis, node_size=node_sizes,
                           node_color=[degrees[node] for node in backbone], cmap="viridis",
                           alpha=0.90, linewidths=0.4, edgecolors="white")
    label_nodes = sorted(backbone, key=lambda node: (-degrees[node], str(node)))[:25]
    nx.draw_networkx_labels(backbone, positions, labels={node: node for node in label_nodes},
                            ax=axis, font_size=7)
    axis.set_title("Co-purchase graph backbone: 100 strongest observed edges", fontsize=16)
    axis.text(0.01, 0.01, "Node size/color = backbone degree; edge width = normalized co-purchase weight",
              transform=axis.transAxes, fontsize=9, color="#475569")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_sku_neighborhood(graph, center_sku, output_path):
    """Plot a labeled, bounded two-hop neighborhood around one SKU."""
    neighborhood, levels = neighborhood_subgraph(graph, center_sku)
    figure, axis = plt.subplots(figsize=(15, 10))
    positions = nx.spring_layout(neighborhood, seed=42, weight="weight", k=0.9)
    colors = {0: "#dc2626", 1: "#2563eb", 2: "#f59e0b"}
    sizes = {0: 900, 1: 480, 2: 260}
    nx.draw_networkx_edges(neighborhood, positions, ax=axis, alpha=0.38,
                           edge_color="#64748b", width=_edge_widths(neighborhood))
    for level in (0, 1, 2):
        nodes = [node for node in neighborhood if levels.get(node) == level]
        nx.draw_networkx_nodes(neighborhood, positions, nodelist=nodes, ax=axis,
                               node_color=colors[level], node_size=sizes[level],
                               edgecolors="white", linewidths=1.0)
    nx.draw_networkx_labels(neighborhood, positions, ax=axis, font_size=8)
    axis.set_title(f"Two-hop co-purchase neighborhood for SKU {center_sku}", fontsize=16)
    axis.text(0.01, 0.01, "Red = query SKU; blue = strongest direct neighbors; orange = selected second hop",
              transform=axis.transAxes, fontsize=9, color="#475569")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_degree_distribution(graph, output_path):
    """Plot degree frequencies, including the isolated-product bar."""
    degrees = pd.Series(dict(graph.degree()).values(), dtype=int)
    frequencies = degrees.value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.bar(frequencies.index, frequencies.values, color="#2563eb", alpha=0.85)
    axis.set_yscale("log")
    axis.set_xlabel("Node degree (number of distinct co-purchased SKUs)")
    axis.set_ylabel("Number of products (log scale)")
    axis.set_title("Product graph degree distribution")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_top_edges(graph, output_path, top_n=20):
    """Plot the product pairs observed together in the most orders."""
    edges = sorted(graph.edges(data=True), key=lambda edge: -edge[2].get("count", 0))[:top_n]
    labels = [f"{left} ↔ {right}" for left, right, _ in edges][::-1]
    counts = [data.get("count", 0) for _, _, data in edges][::-1]
    figure, axis = plt.subplots(figsize=(12, 9))
    axis.barh(labels, counts, color="#0f766e")
    axis.set_xlabel("Orders containing both SKUs")
    axis.set_title(f"Top {top_n} co-purchase edges by raw order count")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_visualizations(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY,
                            center_sku=None):
    """Build the graph and write all visualization and statistics artifacts."""
    orders = load_orders(data_path)
    graph = build_copurchase_graph(orders)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    center_sku = str(center_sku or orders["SKU"].value_counts().index[0])
    safe_sku = re.sub(r"[^\w.-]+", "_", center_sku)
    graph_statistics(graph).to_csv(output_directory / "graph_statistics.csv", index=False,
                                   encoding="utf-8-sig")
    plot_graph_backbone(graph, output_directory / "graph_backbone.png")
    plot_sku_neighborhood(graph, center_sku,
                          output_directory / f"sku_neighborhood_{safe_sku}.png")
    plot_degree_distribution(graph, output_directory / "degree_distribution.png")
    plot_top_edges(graph, output_directory / "top_copurchase_edges.png")
    print(f"Graph visualizations written to {output_directory.resolve()}")
    return graph


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--center-sku", default=None)
    return parser.parse_args()


def main():
    arguments = parse_args()
    generate_visualizations(arguments.data, arguments.output, arguments.center_sku)


if __name__ == "__main__":
    main()
