"""Normalized co-purchase graph and recommendation model."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import sqrt

import networkx as nx
import pandas as pd


def build_copurchase_graph(orders: pd.DataFrame) -> nx.Graph:
    """Build an SKU graph with raw counts and cosine-normalized edge weights."""
    required = {"Order ID", "SKU"}
    if not required.issubset(orders.columns):
        raise ValueError("Orders must include 'Order ID' and 'SKU' columns.")

    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    item_counts = Counter(sku for basket in baskets for sku in basket)
    pair_counts = Counter(pair for basket in baskets for pair in combinations(basket, 2))

    graph = nx.Graph()
    graph.add_nodes_from((sku, {"order_count": count}) for sku, count in item_counts.items())
    for (left, right), count in pair_counts.items():
        normalized = count / sqrt(item_counts[left] * item_counts[right])
        graph.add_edge(left, right, count=count, weight=normalized)
    graph.graph["n_orders"] = len(baskets)
    return graph


class CoPurchaseRecommender:
    def __init__(self):
        self.graph = nx.Graph()
        self.product_catalog = None

    def fit(self, orders, product_catalog=None):
        self.graph = build_copurchase_graph(orders)
        self.product_catalog = product_catalog
        return self

    def recommend(self, sku, top_n=10):
        sku = str(sku)
        columns = [
            "base_sku", "recommended_sku", "copurchase_score",
            "pair_count", "support", "confidence", "jaccard",
        ]
        if sku not in self.graph:
            return pd.DataFrame(columns=columns)

        n_orders = max(self.graph.graph.get("n_orders", 0), 1)
        base_count = self.graph.nodes[sku]["order_count"]
        rows = []
        for other, edge in self.graph[sku].items():
            other_count = self.graph.nodes[other]["order_count"]
            count = edge["count"]
            confidence = count / base_count
            jaccard = count / (base_count + other_count - count)
            rows.append({
                "base_sku": sku,
                "recommended_sku": other,
                "copurchase_score": edge["weight"],
                "pair_count": count,
                "support": count / n_orders,
                "confidence": confidence,
                "jaccard": jaccard,
            })
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values(["copurchase_score", "pair_count"], ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
