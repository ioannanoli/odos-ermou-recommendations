"""Phase 5 link prediction, metadata filtering, and cart recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from math import log
import re

import networkx as nx
import pandas as pd

from src.copurchase_recommender import build_copurchase_graph


class AdamicAdarRecommender:
    """Recommend currently unconnected SKUs through shared graph neighbors."""

    def __init__(self):
        self.graph = nx.Graph()

    def fit(self, orders=None, graph=None):
        """Fit from order lines or reuse an already constructed graph."""
        if graph is None and orders is None:
            raise ValueError("Provide either orders or graph.")
        self.graph = graph.copy() if graph is not None else build_copurchase_graph(orders)
        return self

    def recommend(self, sku, top_n=10):
        """Rank non-neighbor products by the standard Adamic–Adar index."""
        sku = str(sku)
        columns = ["base_sku", "recommended_sku", "adamic_adar_score", "common_neighbors"]
        if sku not in self.graph:
            return pd.DataFrame(columns=columns)
        neighbors = set(self.graph.neighbors(sku))
        candidates = set()
        for neighbor in neighbors:
            candidates.update(self.graph.neighbors(neighbor))
        candidates.difference_update(neighbors | {sku})

        rows = []
        for candidate in candidates:
            common = neighbors.intersection(self.graph.neighbors(candidate))
            score = sum(1.0 / log(self.graph.degree(node)) for node in common
                        if self.graph.degree(node) > 1)
            if score > 0:
                rows.append({
                    "base_sku": sku,
                    "recommended_sku": candidate,
                    "adamic_adar_score": score,
                    "common_neighbors": len(common),
                })
        return (pd.DataFrame(rows, columns=columns)
                .sort_values(["adamic_adar_score", "common_neighbors", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))

    def recommend_cart(self, cart_skus, top_n=10):
        """Aggregate Adamic–Adar evidence from every known product in a cart."""
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cart_set = set(cart)
        scores = {}
        evidence = {}
        for sku in cart:
            for row in self.recommend(sku, top_n=max(top_n * 5, 50)).itertuples():
                if row.recommended_sku in cart_set:
                    continue
                scores[row.recommended_sku] = scores.get(row.recommended_sku, 0.0) + row.adamic_adar_score
                evidence[row.recommended_sku] = evidence.get(row.recommended_sku, 0) + row.common_neighbors
        rows = [{
            "cart_skus": " | ".join(cart),
            "recommended_sku": sku,
            "adamic_adar_score": score,
            "common_neighbors": evidence[sku],
        } for sku, score in scores.items()]
        columns = ["cart_skus", "recommended_sku", "adamic_adar_score", "common_neighbors"]
        return (pd.DataFrame(rows, columns=columns)
                .sort_values(["adamic_adar_score", "common_neighbors", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))


def _allowed_values(value):
    if isinstance(value, str) or not isinstance(value, Iterable):
        return [str(value)]
    return [str(item) for item in value]


def _metadata_matches(value, allowed):
    if pd.isna(value):
        return False
    text = str(value).casefold()
    tokens = {token.strip() for token in re.split(r"[,;|/]", text) if token.strip()}
    return any(candidate.casefold() == text or candidate.casefold() in tokens
               for candidate in _allowed_values(allowed))


def filter_by_metadata(recommendations, product_catalog, metadata_filters):
    """Keep recommendations matching every requested catalog field."""
    if not metadata_filters or recommendations.empty:
        return recommendations.copy()
    missing = set(metadata_filters).difference(product_catalog.columns)
    if missing:
        raise ValueError(f"Unknown metadata filter columns: {sorted(missing)}")
    enriched = recommendations.join(product_catalog, on="recommended_sku")
    mask = pd.Series(True, index=enriched.index)
    for column, allowed in metadata_filters.items():
        mask &= enriched[column].map(lambda value: _metadata_matches(value, allowed))
    return enriched.loc[mask, recommendations.columns].reset_index(drop=True)


class RecommendationEngine:
    """Blend embedding k-NN and Adamic–Adar for a filtered cart query."""

    def __init__(self, product2vec_model, adamic_adar_model, product_catalog,
                 weights=(0.75, 0.25)):
        if len(weights) != 2 or any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("weights must contain two non-negative values with a positive sum.")
        total = sum(weights)
        self.weights = tuple(weight / total for weight in weights)
        self.product2vec_model = product2vec_model
        self.adamic_adar_model = adamic_adar_model
        self.product_catalog = product_catalog

    @staticmethod
    def _normalize(frame, score_column):
        if frame.empty:
            return pd.DataFrame(columns=["recommended_sku", score_column])
        result = frame[["recommended_sku", score_column]].copy()
        minimum, maximum = result[score_column].min(), result[score_column].max()
        if maximum > minimum:
            result[score_column] = (result[score_column] - minimum) / (maximum - minimum)
        else:
            result[score_column] = 1.0 if maximum > 0 else 0.0
        return result

    def recommend(self, cart_skus, top_n=10, metadata_filters=None):
        """Recommend for a cart, optionally requiring catalog metadata values."""
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        if not cart:
            raise ValueError("cart_skus must contain at least one SKU.")
        candidate_count = max(top_n * 10, 100)
        knn = self._normalize(
            self.product2vec_model.recommend_cart(cart, candidate_count), "product2vec_score"
        )
        link = self._normalize(
            self.adamic_adar_model.recommend_cart(cart, candidate_count), "adamic_adar_score"
        )
        if knn.empty and link.empty:
            columns = ["cart_skus", "recommended_sku", "knn_score",
                       "adamic_adar_score", "recommendation_score"]
            return pd.DataFrame(columns=columns)
        combined = knn.merge(link, on="recommended_sku", how="outer")
        for column in ("product2vec_score", "adamic_adar_score"):
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
        combined = filter_by_metadata(combined, self.product_catalog, metadata_filters)
        combined = combined.rename(columns={"product2vec_score": "knn_score"})
        combined["recommendation_score"] = (
            self.weights[0] * combined["knn_score"]
            + self.weights[1] * combined["adamic_adar_score"]
        )
        combined.insert(0, "cart_skus", " | ".join(cart))
        return (combined.sort_values(["recommendation_score", "recommended_sku"],
                                    ascending=[False, True])
                .head(top_n).reset_index(drop=True))
