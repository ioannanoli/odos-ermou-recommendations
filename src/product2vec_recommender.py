"""Node2vec product embeddings trained with Skip-Gram negative sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.copurchase_recommender import build_copurchase_graph


class Product2VecRecommender:
    """Learn structural SKU embeddings from weighted co-purchase walks."""

    def __init__(self, n_components=32, walk_length=8, walks_per_node=1,
                 window_size=3, negative_samples=2, epochs=1,
                 learning_rate=0.025, p=1.0, q=1.0, random_state=42):
        if min(n_components, walk_length, walks_per_node, window_size, epochs) < 1:
            raise ValueError("Embedding and training sizes must be positive.")
        if p <= 0 or q <= 0:
            raise ValueError("Node2vec p and q must be positive.")
        self.n_components = n_components
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.p, self.q = p, q
        self.random_state = random_state
        self.sku_to_index = {}
        self.index_to_sku = {}
        self.embeddings = None
        self.graph = None
        self.product_catalog = None

    def _next_node(self, previous, current, rng):
        neighbors = list(self.graph.neighbors(current))
        if not neighbors:
            return None
        weights = []
        for neighbor in neighbors:
            weight = self.graph[current][neighbor].get("weight", 1.0)
            if previous is None:
                bias = 1.0
            elif neighbor == previous:
                bias = 1.0 / self.p
            elif self.graph.has_edge(previous, neighbor):
                bias = 1.0
            else:
                bias = 1.0 / self.q
            weights.append(weight * bias)
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        return neighbors[rng.choice(len(neighbors), p=probabilities)]

    def generate_walks(self):
        """Generate reproducible second-order weighted Node2vec walks."""
        if self.graph is None:
            raise RuntimeError("Fit the model before generating walks.")
        rng = np.random.default_rng(self.random_state)
        nodes = [node for node in self.graph if self.graph.degree(node) > 0]
        walks = []
        for _ in range(self.walks_per_node):
            for start in rng.permutation(nodes):
                walk = [start]
                while len(walk) < self.walk_length:
                    previous = walk[-2] if len(walk) > 1 else None
                    next_node = self._next_node(previous, walk[-1], rng)
                    if next_node is None:
                        break
                    walk.append(next_node)
                walks.append(walk)
        return walks

    def _positive_pairs(self, walks):
        centers, contexts = [], []
        for walk in walks:
            indices = [self.sku_to_index[str(sku)] for sku in walk]
            for position, center in enumerate(indices):
                left = max(0, position - self.window_size)
                right = min(len(indices), position + self.window_size + 1)
                for offset in range(left, right):
                    if offset != position:
                        centers.append(center)
                        contexts.append(indices[offset])
        return np.asarray(centers, dtype=np.int32), np.asarray(contexts, dtype=np.int32)

    @staticmethod
    def _sigmoid(values):
        return 1.0 / (1.0 + np.exp(-np.clip(values, -15, 15)))

    def _train_skipgram(self, walks, rng):
        centers, positive_contexts = self._positive_pairs(walks)
        vocabulary_size = len(self.sku_to_index)
        scale = 0.5 / max(self.n_components, 1)
        inputs = rng.uniform(-scale, scale, (vocabulary_size, self.n_components))
        outputs = np.zeros_like(inputs)
        if not len(centers):
            return inputs

        frequencies = np.array([
            max(self.graph.nodes[self.index_to_sku[i]].get("order_count", 1), 1)
            for i in range(vocabulary_size)
        ], dtype=float)
        negative_probabilities = frequencies ** 0.75
        negative_probabilities /= negative_probabilities.sum()
        batch_size = 2048

        for epoch in range(self.epochs):
            order = rng.permutation(len(centers))
            rate = self.learning_rate * (1.0 - 0.8 * epoch / max(self.epochs - 1, 1))
            for start in range(0, len(order), batch_size):
                chosen = order[start:start + batch_size]
                batch_centers = centers[chosen]
                batch_positive = positive_contexts[chosen]
                if self.negative_samples:
                    negatives = rng.choice(vocabulary_size,
                        size=(len(chosen), self.negative_samples), p=negative_probabilities)
                    repeated_centers = np.repeat(batch_centers, self.negative_samples + 1)
                    contexts = np.column_stack((batch_positive, negatives)).ravel()
                    labels = np.column_stack((np.ones(len(chosen)), np.zeros_like(negatives))).ravel()
                else:
                    repeated_centers, contexts = batch_centers, batch_positive
                    labels = np.ones(len(chosen))
                center_vectors = inputs[repeated_centers].copy()
                context_vectors = outputs[contexts].copy()
                errors = rate * (labels - self._sigmoid(
                    np.sum(center_vectors * context_vectors, axis=1)))
                np.add.at(inputs, repeated_centers, errors[:, None] * context_vectors)
                np.add.at(outputs, contexts, errors[:, None] * center_vectors)

        embeddings = inputs + outputs
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.divide(embeddings, norms, out=np.zeros_like(embeddings), where=norms > 0)

    def fit(self, orders, product_catalog=None):
        self.graph = build_copurchase_graph(orders)
        self.product_catalog = product_catalog
        skus = sorted(self.graph.nodes)
        self.sku_to_index = {sku: index for index, sku in enumerate(skus)}
        self.index_to_sku = {index: sku for sku, index in self.sku_to_index.items()}
        rng = np.random.default_rng(self.random_state)
        self.embeddings = self._train_skipgram(self.generate_walks(), rng)
        return self

    def recommend(self, sku, top_n=10):
        sku = str(sku)
        columns = ["base_sku", "recommended_sku", "product2vec_score"]
        if (self.embeddings is None or sku not in self.sku_to_index
                or self.graph.degree(sku) == 0):
            return pd.DataFrame(columns=columns)
        index = self.sku_to_index[sku]
        scores = self.embeddings @ self.embeddings[index]
        rows = []
        for other_index in np.argsort(scores)[::-1]:
            other_sku = self.index_to_sku[other_index]
            if other_sku != sku and self.graph.degree(other_sku) > 0:
                rows.append({"base_sku": sku, "recommended_sku": other_sku,
                             "product2vec_score": float(scores[other_index])})
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows, columns=columns)

    def recommend_cart(self, cart_skus, top_n=10):
        """Return exact cosine k-NN recommendations for a multi-product cart."""
        columns = ["cart_skus", "recommended_sku", "product2vec_score"]
        if self.embeddings is None:
            return pd.DataFrame(columns=columns)
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        known = [sku for sku in cart if sku in self.sku_to_index and self.graph.degree(sku) > 0]
        if not known:
            return pd.DataFrame(columns=columns)

        indices = [self.sku_to_index[sku] for sku in known]
        cart_vector = self.embeddings[indices].mean(axis=0)
        norm = np.linalg.norm(cart_vector)
        if norm == 0:
            return pd.DataFrame(columns=columns)
        cart_vector /= norm
        scores = self.embeddings @ cart_vector
        cart_set = set(cart)
        rows = []
        for other_index in np.argsort(scores)[::-1]:
            other_sku = self.index_to_sku[other_index]
            if other_sku not in cart_set and self.graph.degree(other_sku) > 0:
                rows.append({
                    "cart_skus": " | ".join(cart),
                    "recommended_sku": other_sku,
                    "product2vec_score": float(scores[other_index]),
                })
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows, columns=columns)
