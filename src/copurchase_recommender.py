"""Normalized co-purchase graph and recommendation model."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import sqrt

import networkx as nx
import pandas as pd

from src.time_weighting import order_time_weights


GRAPH_WEIGHTINGS = {"cosine", "jaccard", "lift"}


def build_copurchase_graph(orders: pd.DataFrame, weighting: str = "cosine",
                           min_pair_count: int = 1,
                           half_life_months: float | None = None,
                           reference_date=None) -> nx.Graph:
    """Build a configurable, optionally time-weighted SKU co-purchase graph."""
    required = {"Order ID", "SKU"}
    if not required.issubset(orders.columns):
        raise ValueError("Orders must include 'Order ID' and 'SKU' columns.")

    if weighting not in GRAPH_WEIGHTINGS:
        raise ValueError(f"weighting must be one of {sorted(GRAPH_WEIGHTINGS)}")
    if min_pair_count < 1:
        raise ValueError("min_pair_count must be at least one.")

    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    raw_item_counts = Counter(sku for basket in baskets for sku in basket)
    raw_pair_counts = Counter(pair for basket in baskets for pair in combinations(basket, 2))
    if half_life_months is None:
        basket_weights = pd.Series(1.0, index=baskets.index)
    else:
        basket_weights = order_time_weights(
            orders, half_life_months=half_life_months, reference_date=reference_date
        ).reindex(baskets.index)
    weighted_item_counts = defaultdict(float)
    weighted_pair_counts = defaultdict(float)
    for order_id, basket in baskets.items():
        basket_weight = float(basket_weights.loc[order_id])
        for sku in basket:
            weighted_item_counts[sku] += basket_weight
        for pair in combinations(basket, 2):
            weighted_pair_counts[pair] += basket_weight

    graph = nx.Graph()
    graph.add_nodes_from((sku, {
        "order_count": count,
        "weighted_order_count": weighted_item_counts[sku],
    }) for sku, count in raw_item_counts.items())
    effective_orders = float(basket_weights.sum())
    for (left, right), raw_count in raw_pair_counts.items():
        if raw_count < min_pair_count:
            continue
        count = weighted_pair_counts[(left, right)]
        left_count = weighted_item_counts[left]
        right_count = weighted_item_counts[right]
        cosine = count / sqrt(left_count * right_count)
        jaccard = count / (left_count + right_count - count)
        lift = count * effective_orders / (left_count * right_count)
        scores = {"cosine": cosine, "jaccard": jaccard, "lift": lift}
        graph.add_edge(left, right, count=raw_count, weighted_count=count,
                       cosine=cosine, jaccard=jaccard, lift=lift,
                       weight=scores[weighting])
    graph.graph["n_orders"] = len(baskets)
    graph.graph["effective_orders"] = effective_orders
    graph.graph["weighting"] = weighting
    graph.graph["min_pair_count"] = min_pair_count
    graph.graph["half_life_months"] = half_life_months
    graph.graph["reference_date"] = str(
        pd.Timestamp(reference_date) if reference_date is not None else
        pd.to_datetime(orders["Order Date"]).max() if "Order Date" in orders else ""
    )
    return graph


class CoPurchaseRecommender:
    def __init__(self, weighting="cosine", min_pair_count=1,
                 half_life_months=None, reference_date=None, ranking="graph"):
        if ranking not in {"graph", "cosine", "jaccard", "lift", "confidence", "count"}:
            raise ValueError("Unsupported co-purchase ranking score.")
        self.weighting = weighting
        self.min_pair_count = min_pair_count
        self.half_life_months = half_life_months
        self.reference_date = reference_date
        self.ranking = ranking
        self.graph = nx.Graph()
        self.product_catalog = None
        self._cart_cache = {}

    def fit(self, orders, product_catalog=None):
        self.graph = build_copurchase_graph(
            orders, weighting=self.weighting, min_pair_count=self.min_pair_count,
            half_life_months=self.half_life_months, reference_date=self.reference_date,
        )
        self.product_catalog = product_catalog
        self._cart_cache.clear()
        return self

    def _score(self, sku, other, edge):
        if self.ranking == "graph":
            return edge["weight"]
        if self.ranking == "count":
            return edge.get("weighted_count", edge["count"])
        if self.ranking == "confidence":
            base_count = self.graph.nodes[sku].get("weighted_order_count",
                                                   self.graph.nodes[sku]["order_count"])
            return edge.get("weighted_count", edge["count"]) / base_count
        return edge[self.ranking]

    def recommend(self, sku, top_n=10):
        sku = str(sku)
        columns = [
            "base_sku", "recommended_sku", "copurchase_score",
            "pair_count", "support", "confidence", "jaccard",
        ]
        if sku not in self.graph:
            return pd.DataFrame(columns=columns)

        n_orders = max(self.graph.graph.get("effective_orders",
                                            self.graph.graph.get("n_orders", 0)), 1)
        base_count = self.graph.nodes[sku].get(
            "weighted_order_count", self.graph.nodes[sku]["order_count"]
        )
        rows = []
        for other, edge in self.graph[sku].items():
            other_count = self.graph.nodes[other].get(
                "weighted_order_count", self.graph.nodes[other]["order_count"]
            )
            count = edge.get("weighted_count", edge["count"])
            confidence = count / base_count
            jaccard = edge.get("jaccard", count / (base_count + other_count - count))
            rows.append({
                "base_sku": sku,
                "recommended_sku": other,
                "copurchase_score": self._score(sku, other, edge),
                "pair_count": edge["count"],
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

    def recommend_cart(self, cart_skus, top_n=10):
        """Aggregate directional direct co-purchase evidence across a cart."""
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cache_key = (tuple(cart), int(top_n))
        if cache_key in self._cart_cache:
            return self._cart_cache[cache_key].copy()
        cart_set = set(cart)
        scores = defaultdict(float)
        evidence = defaultdict(float)
        for sku in cart:
            if sku not in self.graph:
                continue
            for other, edge in self.graph[sku].items():
                if other not in cart_set:
                    scores[other] += self._score(sku, other, edge)
                    evidence[other] += edge["count"]
        rows = [{
            "cart_skus": " | ".join(cart),
            "recommended_sku": sku,
            "copurchase_score": score,
            "pair_count": evidence[sku],
        } for sku, score in scores.items()]
        columns = ["cart_skus", "recommended_sku", "copurchase_score", "pair_count"]
        result = (pd.DataFrame(rows, columns=columns)
                .sort_values(["copurchase_score", "pair_count", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))
        self._cart_cache[cache_key] = result
        return result.copy()
