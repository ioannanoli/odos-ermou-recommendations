"""Train, persist, and query the development-selected recommendation model."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
from collections.abc import Iterable

import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import create_product_catalog
from src.heterogeneous_graph import HeterogeneousProduct2VecRecommender
from src.metadata_recommender import MetadataSimilarityRecommender
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import AdamicAdarRecommender, HybridRecommendationEngine


DEFAULT_CONFIG_PATH = Path(
    "outputs/improvement_experiments/final/best_dev_configuration.json"
)
REQUIRED_CONFIG_KEYS = {
    "random_seed", "allowed_statuses", "graph", "product2vec",
    "blend_weights", "metadata_field_weights", "architecture",
}


def load_frozen_configuration(path=DEFAULT_CONFIG_PATH):
    """Load and validate the development-selected configuration JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen model configuration not found: {path}")
    configuration = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_CONFIG_KEYS.difference(configuration)
    if missing:
        raise ValueError(f"Frozen configuration is missing keys: {sorted(missing)}")
    if configuration["architecture"] not in {"product_graph", "heterogeneous"}:
        raise ValueError("Unsupported frozen model architecture.")
    if sum(configuration["blend_weights"].values()) <= 0:
        raise ValueError("Frozen blend weights must have a positive total.")
    return configuration


class FinalRecommender:
    """Serving wrapper around the frozen hybrid architecture.

    Model selection remains frozen, but fitting can use all historical orders
    available before deployment.
    """

    def __init__(self, configuration):
        self.configuration = deepcopy(configuration)
        self.engine = None
        self.product_catalog = None
        self.training_summary = {}

    @classmethod
    def from_config_path(cls, path=DEFAULT_CONFIG_PATH):
        return cls(load_frozen_configuration(path))

    def _training_orders(self, orders):
        required = {"Order ID", "Order Date", "SKU"}
        missing = required.difference(orders.columns)
        if missing:
            raise ValueError(f"Training orders are missing columns: {sorted(missing)}")
        allowed = self.configuration.get("allowed_statuses")
        if allowed is None:
            return orders.copy()
        if "Order Status" not in orders:
            raise ValueError("Order Status is required by the frozen status policy.")
        return orders[orders["Order Status"].isin(allowed)].copy()

    def fit(self, orders):
        """Fit the frozen architecture on all supplied historical orders."""
        training = self._training_orders(orders)
        if training.empty:
            raise ValueError("No order lines remain after applying the status policy.")
        catalog = create_product_catalog(training)
        graph_config = self.configuration["graph"]
        copurchase = CoPurchaseRecommender(
            weighting=graph_config["weighting"],
            min_pair_count=int(graph_config["min_pair_count"]),
            half_life_months=graph_config.get("half_life_months"),
            ranking=graph_config.get("ranking", "graph"),
        ).fit(training, catalog)

        p2v_config = {
            **self.configuration["product2vec"],
            "random_state": int(self.configuration["random_seed"]),
        }
        if self.configuration["architecture"] == "heterogeneous":
            product2vec = HeterogeneousProduct2VecRecommender(
                relationship_weights=self.configuration[
                    "heterogeneous_relationship_weights"
                ],
                **p2v_config,
            ).fit(copurchase.graph, catalog)
        else:
            product2vec = Product2VecRecommender(**p2v_config).fit(
                product_catalog=catalog, graph=copurchase.graph
            )

        metadata = MetadataSimilarityRecommender(
            field_weights=self.configuration["metadata_field_weights"]
        ).fit(catalog)
        adamic_adar = None
        if self.configuration["blend_weights"].get("adamic_adar", 0) > 0:
            adamic_adar = AdamicAdarRecommender().fit(graph=copurchase.graph)
        self.engine = HybridRecommendationEngine(
            copurchase, product2vec, adamic_adar, catalog,
            weights=self.configuration["blend_weights"], metadata_model=metadata,
        )
        self.product_catalog = catalog
        dates = pd.to_datetime(training["Order Date"], errors="raise")
        self.training_summary = {
            "architecture": self.configuration["architecture"],
            "order_count": int(training["Order ID"].nunique()),
            "line_count": int(len(training)),
            "catalog_skus": int(len(catalog)),
            "graph_nodes": int(copurchase.graph.number_of_nodes()),
            "graph_edges": int(copurchase.graph.number_of_edges()),
            "training_start": str(dates.min()),
            "training_end": str(dates.max()),
        }
        return self

    def recommend(self, cart_skus, top_n=10, available_skus: Iterable | None = None,
                  metadata_filters=None, enrich=True):
        """Recommend products, optionally restricting results to available SKUs."""
        return self._recommend_with_engine(
            self.engine, cart_skus, top_n, available_skus, metadata_filters, enrich
        )

    def _recommend_with_engine(self, engine, query_skus, top_n,
                               available_skus=None, metadata_filters=None,
                               enrich=True):
        """Run one serving view and apply shared inventory/catalog handling."""
        if self.engine is None:
            raise RuntimeError("Fit or load the final recommender before querying it.")
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        query = list(dict.fromkeys(
            str(sku).strip() for sku in query_skus if str(sku).strip()
        ))
        if not query:
            raise ValueError("The query must contain at least one usable SKU.")
        requested = max(top_n * 10, 100) if available_skus is not None else top_n
        recommendations = engine.recommend(
            query, top_n=requested, metadata_filters=metadata_filters
        )
        if available_skus is not None:
            allowed = {str(sku).strip() for sku in available_skus}
            recommendations = recommendations[
                recommendations["recommended_sku"].isin(allowed)
            ]
        recommendations = recommendations.head(top_n).reset_index(drop=True)
        if enrich and not recommendations.empty:
            recommendations = recommendations.join(
                self.product_catalog, on="recommended_sku"
            )
        return recommendations

    def _view_engine(self, weights):
        """Create a ranking view over the already-fitted component models."""
        if self.engine is None:
            raise RuntimeError("Fit or load the final recommender before querying it.")
        return HybridRecommendationEngine(
            self.engine.copurchase_model,
            self.engine.product2vec_model,
            self.engine.adamic_adar_model,
            self.product_catalog,
            weights=weights,
            metadata_model=self.engine.metadata_model,
        )

    def recommend_frequently_bought_together(
        self, sku, top_n=10, available_skus: Iterable | None = None,
        enrich=True,
    ):
        """Rank direct historical complements for a single product page."""
        engine = self._view_engine({"copurchase": 1.0})
        return self._recommend_with_engine(
            engine, [sku], top_n, available_skus, metadata_filters=None,
            enrich=enrich,
        )

    def recommend_similar(
        self, sku, top_n=10, available_skus: Iterable | None = None,
        metadata_filters=None, enrich=True,
    ):
        """Rank product substitutes using embeddings and structured content."""
        engine = self._view_engine({"product2vec": 0.5, "metadata": 0.5})
        return self._recommend_with_engine(
            engine, [sku], top_n, available_skus, metadata_filters, enrich,
        )

    def save(self, path):
        """Persist a locally trained model. Only load trusted pickle files."""
        if self.engine is None:
            raise RuntimeError("Fit the final recommender before saving it.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path):
        """Load a trusted model artifact created by :meth:`save`."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trained model not found: {path}")
        with path.open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls):
            raise TypeError("The model artifact is not a FinalRecommender.")
        return model
