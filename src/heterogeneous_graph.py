"""Experimental metadata-enriched product graph and SKU-only embeddings."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt

import networkx as nx
import numpy as np
import pandas as pd

from src.metadata_recommender import metadata_tokens
from src.product2vec_recommender import Product2VecRecommender


METADATA_RELATIONS = {
    "category": "Κατηγορίες προϊόντων",
    "brand": "Μάρκες",
    "age": "Προϊόν Ηλικία",
    "hero": "Προϊόν Ήρωας",
}
DEFAULT_RELATIONSHIP_WEIGHTS = {
    "copurchase": 1.0, "category": 0.10, "brand": 0.10,
    "age": 0.10, "hero": 0.10,
}


def product_node(sku):
    return f"product::{sku}"


def build_heterogeneous_graph(product_graph, product_catalog,
                              relationship_weights=None):
    """Add typed metadata nodes with inverse-sqrt hub down-weighting."""
    weights = {**DEFAULT_RELATIONSHIP_WEIGHTS, **(relationship_weights or {})}
    if any(value < 0 for value in weights.values()):
        raise ValueError("Relationship weights must be non-negative.")
    missing = set(METADATA_RELATIONS.values()).difference(product_catalog.columns)
    if missing:
        raise ValueError(f"Metadata columns are missing: {sorted(missing)}")
    graph = nx.Graph()
    for sku, attributes in product_graph.nodes(data=True):
        graph.add_node(product_node(sku), **attributes, node_type="product", sku=str(sku))
    for left, right, attributes in product_graph.edges(data=True):
        edge = dict(attributes)
        edge["weight"] = float(attributes.get("weight", 1.0)) * weights["copurchase"]
        edge["relation"] = "copurchase"
        graph.add_edge(product_node(left), product_node(right), **edge)

    memberships = defaultdict(set)
    for raw_sku, row in product_catalog.iterrows():
        sku = str(raw_sku)
        if sku not in product_graph:
            continue
        for relation, column in METADATA_RELATIONS.items():
            for token in metadata_tokens(row[column]):
                memberships[(relation, token)].add(sku)
    for (relation, token), skus in memberships.items():
        if not skus or weights[relation] == 0:
            continue
        metadata_id = f"{relation}::{token}"
        graph.add_node(metadata_id, node_type=relation, value=token,
                       order_count=1, weighted_order_count=1.0)
        edge_weight = weights[relation] / sqrt(len(skus))
        for sku in skus:
            graph.add_edge(product_node(sku), metadata_id, weight=edge_weight,
                           relation=f"has_{relation}", count=1, weighted_count=1.0)
    graph.graph.update({"relationship_weights": weights, "heterogeneous": True})
    return graph


class HeterogeneousProduct2VecRecommender:
    """Train ordinary weighted walks on the mixed graph, returning SKUs only."""

    def __init__(self, relationship_weights=None, **product2vec_config):
        self.relationship_weights = {
            **DEFAULT_RELATIONSHIP_WEIGHTS, **(relationship_weights or {})
        }
        self.product2vec_config = product2vec_config
        self.model = Product2VecRecommender(**product2vec_config)
        self.graph = nx.Graph()
        self.heterogeneous_graph = nx.Graph()
        self.product_catalog = None
        self._cart_cache = {}

    def fit(self, product_graph, product_catalog):
        self.graph = product_graph.copy()
        self.product_catalog = product_catalog
        self._cart_cache.clear()
        self.heterogeneous_graph = build_heterogeneous_graph(
            self.graph, product_catalog, self.relationship_weights
        )
        self.model.fit(product_catalog=product_catalog, graph=self.heterogeneous_graph)
        return self

    def recommend_cart(self, cart_skus, top_n=10):
        columns = ["cart_skus", "recommended_sku", "product2vec_score"]
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cache_key = (tuple(cart), int(top_n))
        if cache_key in self._cart_cache:
            return self._cart_cache[cache_key].copy()
        known_nodes = [product_node(sku) for sku in cart
                       if product_node(sku) in self.model.sku_to_index]
        if not known_nodes or self.model.embeddings is None:
            return pd.DataFrame(columns=columns)
        indices = [self.model.sku_to_index[node] for node in known_nodes]
        vector = self.model.embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return pd.DataFrame(columns=columns)
        scores = self.model.embeddings @ (vector / norm)
        cart_set = set(cart)
        rows = []
        for index in np.argsort(scores)[::-1]:
            node = self.model.index_to_sku[index]
            if not node.startswith("product::"):
                continue
            sku = node.split("::", 1)[1]
            if sku not in cart_set and self.graph.degree(sku) > 0:
                rows.append({"cart_skus": " | ".join(cart), "recommended_sku": sku,
                             "product2vec_score": float(scores[index])})
            if len(rows) == top_n:
                break
        result = pd.DataFrame(rows, columns=columns)
        self._cart_cache[cache_key] = result
        return result.copy()

    def recommend(self, sku, top_n=10):
        result = self.recommend_cart([sku], top_n)
        if result.empty:
            return pd.DataFrame(columns=["base_sku", "recommended_sku",
                                         "product2vec_score"])
        result = result.drop(columns="cart_skus")
        result.insert(0, "base_sku", str(sku))
        return result
