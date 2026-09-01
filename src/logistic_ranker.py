"""Interpretable candidate features and dependency-free logistic ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


METADATA_MATCH_NAMES = (
    "same_category", "same_brand", "same_age", "same_hero", "same_gender",
)
FEATURE_NAMES = (
    "copurchase_score", "product2vec_score", "metadata_score", "text_score",
    "log_pair_count", "copurchase_confidence",
    "log_seed_order_count", "log_candidate_order_count",
    "log_candidate_graph_degree", "log_seed_recency_days",
    "log_candidate_recency_days", *METADATA_MATCH_NAMES,
    "source_copurchase", "source_product2vec", "source_metadata", "source_text",
    "candidate_source_count", "missing_behavioral", "missing_content",
    "seed_behavioral_reliability", "copurchase_x_reliability",
    "product2vec_x_reliability", "metadata_x_content_need",
    "text_x_content_need",
)


def _source_set(value):
    return {source for source in str(value).split(" | ") if source}


class CandidateFeatureBuilder:
    """Create past-only ranking features for one fitted temporal snapshot."""

    def __init__(self, model, training_orders, reliability_scale=10.0):
        if model.engine is None:
            raise RuntimeError("The candidate model must be fitted first.")
        if reliability_scale <= 0:
            raise ValueError("reliability_scale must be positive.")
        self.model = model
        self.engine = model.engine
        self.graph = self.engine.copurchase_model.graph
        self.catalog = model.product_catalog
        self.reliability_scale = float(reliability_scale)
        dated = training_orders.copy()
        dated["Order Date"] = pd.to_datetime(dated["Order Date"], errors="raise")
        self.reference_date = dated["Order Date"].max()
        self.last_order_dates = dated.groupby("SKU")["Order Date"].max().to_dict()
        self.metadata_fields = tuple(self.engine.metadata_model.field_weights)
        if len(self.metadata_fields) != len(METADATA_MATCH_NAMES):
            raise ValueError("The logistic feature builder expects five metadata fields.")

    def _order_count(self, sku):
        if sku not in self.graph:
            return 0
        return int(self.graph.nodes[sku].get("order_count", 0))

    def _recency_days(self, sku):
        last_date = self.last_order_dates.get(sku)
        if last_date is None:
            return 3650.0
        return max(float((self.reference_date - pd.Timestamp(last_date)).days), 0.0)

    def transform(self, seed_sku, candidate_pool):
        """Return one numerical feature row per candidate in pool order."""
        seed = str(seed_sku)
        if candidate_pool.empty:
            return np.empty((0, len(FEATURE_NAMES)), dtype=float)
        required = {
            "recommended_sku", "copurchase_score", "product2vec_score",
            "metadata_score", "text_score", "candidate_sources",
            "candidate_source_count",
        }
        missing = required.difference(candidate_pool.columns)
        if missing:
            raise ValueError(f"Candidate pool is missing feature columns: {sorted(missing)}")

        seed_count = self._order_count(seed)
        reliability = seed_count / (seed_count + self.reliability_scale)
        content_need = 1.0 - reliability
        seed_recency = np.log1p(self._recency_days(seed))
        seed_metadata = self.engine.metadata_model._features.get(seed, {})
        rows = []
        for candidate_row in candidate_pool.itertuples(index=False):
            candidate = str(candidate_row.recommended_sku)
            sources = _source_set(candidate_row.candidate_sources)
            edge = self.graph.get_edge_data(seed, candidate, default={})
            pair_count = float(edge.get("count", 0.0))
            seed_weighted_count = (
                float(self.graph.nodes[seed].get(
                    "weighted_order_count", self.graph.nodes[seed].get("order_count", 0)
                )) if seed in self.graph else 0.0
            )
            confidence = (
                float(edge.get("weighted_count", pair_count)) / seed_weighted_count
                if seed_weighted_count > 0 else 0.0
            )
            candidate_count = self._order_count(candidate)
            candidate_degree = self.graph.degree(candidate) if candidate in self.graph else 0
            candidate_metadata = self.engine.metadata_model._features.get(candidate, {})
            metadata_matches = [
                float(bool(seed_metadata.get(field, set()) &
                           candidate_metadata.get(field, set())))
                for field in self.metadata_fields
            ]
            copurchase = float(candidate_row.copurchase_score)
            product2vec = float(candidate_row.product2vec_score)
            metadata = float(candidate_row.metadata_score)
            text = float(candidate_row.text_score)
            source_values = [
                float(signal in sources)
                for signal in ("copurchase", "product2vec", "metadata", "text")
            ]
            missing_behavioral = float(
                "copurchase" not in sources and "product2vec" not in sources
            )
            missing_content = float(
                "metadata" not in sources and "text" not in sources
            )
            rows.append([
                copurchase, product2vec, metadata, text,
                np.log1p(pair_count), confidence,
                np.log1p(seed_count), np.log1p(candidate_count),
                np.log1p(candidate_degree), seed_recency,
                np.log1p(self._recency_days(candidate)), *metadata_matches,
                *source_values, float(candidate_row.candidate_source_count),
                missing_behavioral, missing_content, reliability,
                copurchase * reliability, product2vec * reliability,
                metadata * content_need, text * content_need,
            ])
        return np.asarray(rows, dtype=float)


class RegularizedLogisticRanker:
    """L2-regularized pointwise logistic regression solved with Newton steps."""

    def __init__(self, l2=1.0, max_iterations=50, tolerance=1e-7):
        if l2 < 0 or max_iterations < 1 or tolerance <= 0:
            raise ValueError("Invalid logistic-regression optimization settings.")
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.feature_names = FEATURE_NAMES
        self.feature_mean = None
        self.feature_scale = None
        self.coefficients = None
        self.intercept = 0.0
        self.iterations = 0

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-values))

    def fit(self, features, labels, sample_weight=None):
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("features have the wrong shape for the logistic ranker.")
        if len(features) != len(labels) or len(features) == 0:
            raise ValueError("features and labels must be non-empty and aligned.")
        if set(np.unique(labels)).difference({0.0, 1.0}) or len(np.unique(labels)) < 2:
            raise ValueError("labels must contain both zero and one.")

        self.feature_mean = features.mean(axis=0)
        scale = features.std(axis=0)
        self.feature_scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (features - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        base_weight = (
            np.ones(len(labels), dtype=float) if sample_weight is None
            else np.asarray(sample_weight, dtype=float)
        )
        if base_weight.shape != labels.shape or np.any(base_weight <= 0):
            raise ValueError("sample_weight must contain one positive value per row.")
        positive_mass = base_weight[labels == 1].sum()
        negative_mass = base_weight[labels == 0].sum()
        balanced_weight = base_weight.copy()
        balanced_weight[labels == 1] *= 0.5 / positive_mass
        balanced_weight[labels == 0] *= 0.5 / negative_mass

        parameters = np.zeros(design.shape[1], dtype=float)
        regularization = np.r_[0.0, np.full(features.shape[1], self.l2)]
        for iteration in range(1, self.max_iterations + 1):
            probabilities = self._sigmoid(design @ parameters)
            gradient = design.T @ (balanced_weight * (probabilities - labels))
            gradient += regularization * parameters
            curvature = balanced_weight * probabilities * (1.0 - probabilities)
            hessian = design.T @ (design * curvature[:, None])
            hessian += np.diag(regularization + 1e-10)
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            parameters -= step
            self.iterations = iteration
            if np.max(np.abs(step)) < self.tolerance:
                break
        self.intercept = float(parameters[0])
        self.coefficients = parameters[1:]
        return self

    def predict_proba(self, features):
        if self.coefficients is None:
            raise RuntimeError("Fit the logistic ranker before prediction.")
        features = np.asarray(features, dtype=float)
        standardized = (features - self.feature_mean) / self.feature_scale
        return self._sigmoid(self.intercept + standardized @ self.coefficients)

    def coefficient_frame(self):
        if self.coefficients is None:
            raise RuntimeError("Fit the logistic ranker first.")
        return (pd.DataFrame({
            "feature": self.feature_names,
            "standardized_coefficient": self.coefficients,
            "absolute_coefficient": np.abs(self.coefficients),
        }).sort_values("absolute_coefficient", ascending=False)
          .reset_index(drop=True))
