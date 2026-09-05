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
        self._cart_cache = {}

    def fit(self, orders=None, graph=None):
        """Fit from order lines or reuse an already constructed graph."""
        if graph is None and orders is None:
            raise ValueError("Provide either orders or graph.")
        self.graph = graph.copy() if graph is not None else build_copurchase_graph(orders)
        self._cart_cache.clear()
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
        cache_key = (tuple(cart), int(top_n))
        if cache_key in self._cart_cache:
            return self._cart_cache[cache_key].copy()
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
        result = (pd.DataFrame(rows, columns=columns)
                .sort_values(["adamic_adar_score", "common_neighbors", "recommended_sku"],
                             ascending=[False, False, True])
                .head(top_n).reset_index(drop=True))
        self._cart_cache[cache_key] = result
        return result.copy()


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


class HybridRecommendationEngine:
    """Blend direct, embedding, link-prediction, and metadata signals."""

    SCORE_COLUMNS = {
        "copurchase": "copurchase_score",
        "product2vec": "product2vec_score",
        "adamic_adar": "adamic_adar_score",
        "metadata": "metadata_score",
        "text": "text_score",
    }

    def __init__(self, copurchase_model, product2vec_model, adamic_adar_model,
                 product_catalog, weights=None, metadata_model=None,
                 text_model=None, candidate_generation=None):
        weights = dict(weights or {"product2vec": 0.75, "adamic_adar": 0.25})
        unknown = set(weights).difference(self.SCORE_COLUMNS)
        if unknown or any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError(f"Invalid hybrid weights; supported signals: {sorted(self.SCORE_COLUMNS)}")
        if weights.get("metadata", 0) > 0 and metadata_model is None:
            raise ValueError("A metadata_model is required when metadata weight is positive.")
        if weights.get("text", 0) > 0 and text_model is None:
            raise ValueError("A text_model is required when text weight is positive.")
        total = sum(weights.values())
        self.weights = {name: weights.get(name, 0.0) / total for name in self.SCORE_COLUMNS}
        self.copurchase_model = copurchase_model
        self.product2vec_model = product2vec_model
        self.adamic_adar_model = adamic_adar_model
        self.metadata_model = metadata_model
        self.text_model = text_model
        self.product_catalog = product_catalog
        self.graph = product2vec_model.graph
        self.candidate_generation = self._validate_candidate_generation(
            candidate_generation
        )

    @classmethod
    def _validate_candidate_generation(cls, configuration):
        """Validate retrieval settings without changing the legacy default."""
        configuration = dict(configuration or {})
        supported = {
            "minimum_per_source", "multiplier", "maximum_per_source",
            "source_limits", "track_sources",
        }
        unknown = set(configuration).difference(supported)
        if unknown:
            raise ValueError(
                f"Unknown candidate-generation settings: {sorted(unknown)}"
            )
        minimum = int(configuration.get("minimum_per_source", 100))
        multiplier = int(configuration.get("multiplier", 10))
        maximum = configuration.get("maximum_per_source")
        maximum = None if maximum is None else int(maximum)
        source_limits = dict(configuration.get("source_limits", {}))
        unknown_sources = set(source_limits).difference(cls.SCORE_COLUMNS)
        if minimum < 1 or multiplier < 1:
            raise ValueError(
                "minimum_per_source and multiplier must be positive integers."
            )
        if maximum is not None and maximum < minimum:
            raise ValueError(
                "maximum_per_source must be at least minimum_per_source."
            )
        if unknown_sources or any(int(value) < 1 for value in source_limits.values()):
            raise ValueError(
                "source_limits must contain positive limits for supported signals."
            )
        return {
            "minimum_per_source": minimum,
            "multiplier": multiplier,
            "maximum_per_source": maximum,
            "source_limits": {
                signal: int(value) for signal, value in source_limits.items()
            },
            "track_sources": bool(configuration.get("track_sources", False)),
        }

    def _source_limit(self, signal, requested_top_n):
        """Return the retrieval depth for one independent candidate source."""
        explicit = self.candidate_generation["source_limits"].get(signal)
        if explicit is not None:
            return explicit
        limit = max(
            int(requested_top_n) * self.candidate_generation["multiplier"],
            self.candidate_generation["minimum_per_source"],
        )
        maximum = self.candidate_generation["maximum_per_source"]
        return min(limit, maximum) if maximum is not None else limit

    @staticmethod
    def _normalized_scores(frame, score_column):
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["recommended_sku", score_column])
        result = frame[["recommended_sku", score_column]].copy()
        result[score_column] = pd.to_numeric(result[score_column], errors="coerce").fillna(0.0)
        minimum, maximum = result[score_column].min(), result[score_column].max()
        if maximum > minimum:
            result[score_column] = (result[score_column] - minimum) / (maximum - minimum)
        else:
            result[score_column] = 1.0 if maximum > 0 else 0.0
        return result

    def candidate_pool(self, cart_skus, top_n=10, metadata_filters=None):
        """Retrieve, merge, score, and return the full multi-source pool.

        ``top_n`` describes the downstream ranking request. Retrieval depth is
        controlled separately by ``candidate_generation``. The returned frame
        is not truncated, which lets offline evaluation measure whether a
        hidden target was retrieved before judging its final rank.
        """
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        if not cart:
            raise ValueError("cart_skus must contain at least one SKU.")
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        models = {
            "copurchase": self.copurchase_model,
            "product2vec": self.product2vec_model,
            "adamic_adar": self.adamic_adar_model,
            "metadata": self.metadata_model,
            "text": self.text_model,
        }
        combined = None
        for signal, score_column in self.SCORE_COLUMNS.items():
            model = models[signal]
            if model is None or self.weights[signal] == 0:
                continue
            source_limit = self._source_limit(signal, top_n)
            frame = self._normalized_scores(
                model.recommend_cart(cart, source_limit), score_column
            )
            frame[f"source_{signal}"] = True
            combined = frame if combined is None else combined.merge(
                frame, on="recommended_sku", how="outer"
            )
        output_columns = ["cart_skus", "recommended_sku", *self.SCORE_COLUMNS.values(),
                          "recommendation_score"]
        if combined is None or combined.empty:
            if self.candidate_generation["track_sources"]:
                output_columns.extend(["candidate_sources", "candidate_source_count"])
            return pd.DataFrame(columns=output_columns)
        for score_column in self.SCORE_COLUMNS.values():
            if score_column not in combined:
                combined[score_column] = 0.0
            combined[score_column] = pd.to_numeric(
                combined[score_column], errors="coerce"
            ).fillna(0.0)
        combined = filter_by_metadata(combined, self.product_catalog, metadata_filters)
        combined["recommendation_score"] = sum(
            self.weights[signal] * combined[score_column]
            for signal, score_column in self.SCORE_COLUMNS.items()
        )
        if self.candidate_generation["track_sources"]:
            source_columns = []
            for signal in self.SCORE_COLUMNS:
                source_column = f"source_{signal}"
                if source_column not in combined:
                    combined[source_column] = False
                combined[source_column] = combined[source_column].map(
                    lambda value: bool(value) if pd.notna(value) else False
                )
                source_columns.append(source_column)
            combined["candidate_sources"] = combined.apply(
                lambda row: " | ".join(
                    signal for signal in self.SCORE_COLUMNS
                    if row[f"source_{signal}"]
                ),
                axis=1,
            )
            combined["candidate_source_count"] = combined[source_columns].sum(axis=1)
            output_columns.extend(["candidate_sources", "candidate_source_count"])
        combined.insert(0, "cart_skus", " | ".join(cart))
        return (combined.sort_values(["recommendation_score", "recommended_sku"],
                                    ascending=[False, True])
                .reset_index(drop=True)[output_columns])

    def recommend(self, cart_skus, top_n=10, metadata_filters=None):
        """Rank the top products from the independently retrieved pool."""
        return (self.candidate_pool(cart_skus, top_n, metadata_filters)
                .head(top_n).reset_index(drop=True))
