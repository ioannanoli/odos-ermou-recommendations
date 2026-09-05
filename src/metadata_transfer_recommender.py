"""Content-bridged transfer of co-purchase complements to sparse products."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
import re

import networkx as nx
import pandas as pd

from src.copurchase_recommender import build_copurchase_graph
from src.recommendation_engine import filter_by_metadata


DEFAULT_FIELD_WEIGHTS = {
    "Κατηγορίες προϊόντων": 0.35,
    "Προϊόν Ηλικία": 0.30,
    "Προϊόν Ήρωας": 0.15,
    "Μάρκες": 0.10,
    "Προϊόν Φύλλο": 0.05,
    "Product Name": 0.05,
}


def _metadata_tokens(value, product_name=False):
    """Normalize a catalog value into comparable, non-missing tokens."""
    if pd.isna(value) or str(value).strip().casefold() in {"", "unknown", "nan", "none"}:
        return set()
    pattern = r"[^\w]+" if product_name else r"[,;|/>]+"
    tokens = {token.strip().casefold() for token in re.split(pattern, str(value))
              if len(token.strip()) >= (3 if product_name else 1)}
    return tokens


class MetadataTransferRecommender:
    """Transfer complements through metadata-similar established products."""

    def __init__(self, field_weights=None, metadata_neighbors=20,
                 min_similarity=0.20, reliability_lambda=5.0,
                 substitution_penalty=0.75):
        self.field_weights = dict(field_weights or DEFAULT_FIELD_WEIGHTS)
        if not self.field_weights or any(weight < 0 for weight in self.field_weights.values()):
            raise ValueError("field_weights must contain non-negative values.")
        total = sum(self.field_weights.values())
        if total <= 0:
            raise ValueError("field_weights must have a positive total.")
        self.field_weights = {field: weight / total for field, weight in self.field_weights.items()}
        self.metadata_neighbors = metadata_neighbors
        self.min_similarity = min_similarity
        self.reliability_lambda = reliability_lambda
        self.substitution_penalty = substitution_penalty
        self.graph = nx.Graph()
        self.product_catalog = None
        self._features = {}
        self._inverted_index = defaultdict(set)
        self._neighbor_cache = {}

    def fit(self, orders=None, product_catalog=None, graph=None):
        """Index catalog metadata and fit or reuse the co-purchase graph."""
        if product_catalog is None:
            raise ValueError("product_catalog is required.")
        missing = set(self.field_weights).difference(product_catalog.columns)
        if missing:
            raise ValueError(f"Metadata columns are missing: {sorted(missing)}")
        if graph is None and orders is None:
            raise ValueError("Provide either orders or graph.")
        self.graph = graph.copy() if graph is not None else build_copurchase_graph(orders)
        self.product_catalog = product_catalog.copy()
        self._features.clear()
        self._inverted_index.clear()
        self._neighbor_cache.clear()
        for raw_sku, row in self.product_catalog.iterrows():
            sku = str(raw_sku)
            features = {
                field: _metadata_tokens(row[field], product_name=(field == "Product Name"))
                for field in self.field_weights
            }
            self._features[sku] = features
            for field, tokens in features.items():
                for token in tokens:
                    self._inverted_index[(field, token)].add(sku)
        return self

    def metadata_similarity(self, left_sku, right_sku):
        """Return weighted field-wise Jaccard similarity between two SKUs."""
        left, right = self._features.get(str(left_sku)), self._features.get(str(right_sku))
        if left is None or right is None:
            return 0.0
        score = 0.0
        for field, weight in self.field_weights.items():
            left_tokens, right_tokens = left[field], right[field]
            union = left_tokens | right_tokens
            if union:
                score += weight * len(left_tokens & right_tokens) / len(union)
        return score

    def _reliability(self, sku):
        count = self.graph.nodes[sku].get("order_count", 0) if sku in self.graph else 0
        return count / (count + self.reliability_lambda) if count else 0.0

    def similar_products(self, sku, top_n=None):
        """Find established metadata neighbors without scanning the full catalog."""
        sku = str(sku)
        limit = top_n or self.metadata_neighbors
        cache_key = (sku, limit)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key].copy()
        if sku not in self._features:
            columns = ["base_sku", "similar_sku", "metadata_similarity", "reliability", "bridge_score"]
            return pd.DataFrame(columns=columns)
        candidates = set()
        for field, tokens in self._features[sku].items():
            for token in tokens:
                candidates.update(self._inverted_index[(field, token)])
        candidates.discard(sku)
        rows = []
        for candidate in candidates:
            similarity = self.metadata_similarity(sku, candidate)
            reliability = self._reliability(candidate)
            if similarity >= self.min_similarity and reliability > 0 and self.graph.degree(candidate) > 0:
                rows.append({
                    "base_sku": sku,
                    "similar_sku": candidate,
                    "metadata_similarity": similarity,
                    "reliability": reliability,
                    "bridge_score": similarity * reliability,
                })
        columns = ["base_sku", "similar_sku", "metadata_similarity", "reliability", "bridge_score"]
        result = (pd.DataFrame(rows, columns=columns)
                  .sort_values(["bridge_score", "metadata_similarity", "similar_sku"],
                               ascending=[False, False, True])
                  .head(limit).reset_index(drop=True))
        self._neighbor_cache[cache_key] = result
        return result.copy()

    def recommend(self, sku, top_n=10):
        """Blend direct evidence with complements transferred through bridges."""
        sku = str(sku)
        direct_scores = {}
        if sku in self.graph:
            direct_scores = {neighbor: edge.get("weight", 0.0)
                             for neighbor, edge in self.graph[sku].items()}
        transfer_scores = defaultdict(float)
        bridge_counts = defaultdict(int)
        for bridge in self.similar_products(sku).itertuples():
            for complement, edge in self.graph[bridge.similar_sku].items():
                if complement == sku:
                    continue
                complement_factor = max(
                    0.0,
                    1.0 - self.substitution_penalty * self.metadata_similarity(sku, complement),
                )
                contribution = (bridge.metadata_similarity * bridge.reliability
                                * edge.get("weight", 0.0) * complement_factor)
                if contribution > 0:
                    transfer_scores[complement] += contribution
                    bridge_counts[complement] += 1
        query_reliability = self._reliability(sku)
        candidates = (set(direct_scores) | set(transfer_scores)) - {sku}
        rows = [{
            "base_sku": sku,
            "recommended_sku": candidate,
            "direct_copurchase_score": direct_scores.get(candidate, 0.0),
            "transferred_complement_score": transfer_scores.get(candidate, 0.0),
            "metadata_bridges": bridge_counts.get(candidate, 0),
            "metadata_transfer_score": (
                query_reliability * direct_scores.get(candidate, 0.0)
                + (1.0 - query_reliability) * transfer_scores.get(candidate, 0.0)
            ),
        } for candidate in candidates]
        columns = ["base_sku", "recommended_sku", "direct_copurchase_score",
                   "transferred_complement_score", "metadata_bridges", "metadata_transfer_score"]
        return (pd.DataFrame(rows, columns=columns)
                .sort_values(["metadata_transfer_score", "metadata_bridges", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))

    def recommend_cart(self, cart_skus, top_n=10):
        """Sum direct and transferred evidence across all products in a cart."""
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cart_set = set(cart)
        aggregates = defaultdict(lambda: [0.0, 0.0, 0])
        for sku in cart:
            for row in self.recommend(sku, top_n=max(top_n * 5, 50)).itertuples():
                if row.recommended_sku not in cart_set:
                    aggregates[row.recommended_sku][0] += row.direct_copurchase_score
                    aggregates[row.recommended_sku][1] += row.transferred_complement_score
                    aggregates[row.recommended_sku][2] += row.metadata_bridges
        rows = [{
            "cart_skus": " | ".join(cart),
            "recommended_sku": candidate,
            "direct_copurchase_score": values[0],
            "transferred_complement_score": values[1],
            "metadata_bridges": values[2],
            "metadata_transfer_score": values[0] + values[1],
        } for candidate, values in aggregates.items()]
        columns = ["cart_skus", "recommended_sku", "direct_copurchase_score",
                   "transferred_complement_score", "metadata_bridges", "metadata_transfer_score"]
        return (pd.DataFrame(rows, columns=columns)
                .sort_values(["metadata_transfer_score", "metadata_bridges", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))


class MetadataEnhancedEngine:
    """Blend the existing cart engine with transferred complement evidence."""

    def __init__(self, base_engine, transfer_model, transfer_weight=0.30):
        if not 0 <= transfer_weight <= 1:
            raise ValueError("transfer_weight must be between zero and one.")
        self.base_engine = base_engine
        self.transfer_model = transfer_model
        self.transfer_weight = transfer_weight
        self.product2vec_model = base_engine.product2vec_model
        self.product_catalog = base_engine.product_catalog

    @staticmethod
    def _normalize(series):
        series = pd.to_numeric(series, errors="coerce").fillna(0.0)
        minimum, maximum = series.min(), series.max()
        if maximum > minimum:
            return (series - minimum) / (maximum - minimum)
        return pd.Series(1.0 if maximum > 0 else 0.0, index=series.index)

    def recommend(self, cart_skus, top_n=10, metadata_filters=None):
        candidate_count = max(top_n * 10, 100)
        base = self.base_engine.recommend(cart_skus, candidate_count)
        transfer = self.transfer_model.recommend_cart(cart_skus, candidate_count)
        base = base[["recommended_sku", "recommendation_score"]].rename(
            columns={"recommendation_score": "base_recommendation_score"}
        ) if not base.empty else pd.DataFrame(
            columns=["recommended_sku", "base_recommendation_score"]
        )
        transfer = transfer[["recommended_sku", "transferred_complement_score",
                             "metadata_bridges"]] if not transfer.empty else pd.DataFrame(
            columns=["recommended_sku", "transferred_complement_score", "metadata_bridges"]
        )
        if base.empty and transfer.empty:
            return pd.DataFrame(columns=["cart_skus", "recommended_sku",
                                         "base_recommendation_score", "transferred_complement_score",
                                         "metadata_bridges", "recommendation_score"])
        combined = base.merge(transfer, on="recommended_sku", how="outer")
        for column in ("base_recommendation_score", "transferred_complement_score",
                       "metadata_bridges"):
            combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)
        combined["transferred_complement_score"] = self._normalize(
            combined["transferred_complement_score"]
        )
        combined = filter_by_metadata(combined, self.product_catalog, metadata_filters)
        combined["recommendation_score"] = (
            (1.0 - self.transfer_weight) * combined["base_recommendation_score"]
            + self.transfer_weight * combined["transferred_complement_score"]
        )
        combined.insert(0, "cart_skus", " | ".join(map(str, cart_skus)))
        return (combined.sort_values(["recommendation_score", "recommended_sku"],
                                    ascending=[False, True])
                .head(top_n).reset_index(drop=True))
