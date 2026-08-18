"""Phase 6 leave-one-out evaluation and qualitative error analysis."""

from __future__ import annotations

from math import log
import re

import numpy as np
import pandas as pd


def softmax_cross_entropy(logits, target_index):
    """Return stable categorical cross-entropy for one vector of raw logits."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("logits must be a non-empty one-dimensional sequence.")
    if not 0 <= target_index < logits.size:
        raise IndexError("target_index is outside the logits vector.")
    shifted = logits - logits.max()
    log_sum_exp = np.log(np.exp(shifted).sum())
    return float(log_sum_exp - shifted[target_index])


def leave_one_out_queries(orders, known_skus, max_queries=None, random_state=42):
    """Create reproducible cart/hidden-target queries from multi-item orders."""
    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    eligible = [(str(order_id), [sku for sku in basket if sku in known_skus])
                for order_id, basket in baskets.items()]
    eligible = [(order_id, basket) for order_id, basket in eligible if len(basket) >= 2]
    rng = np.random.default_rng(random_state)
    if max_queries is not None and len(eligible) > max_queries:
        chosen = np.sort(rng.choice(len(eligible), size=max_queries, replace=False))
        eligible = [eligible[index] for index in chosen]

    queries = []
    for order_id, basket in eligible:
        target_index = int(rng.integers(len(basket)))
        target = basket[target_index]
        cart = basket[:target_index] + basket[target_index + 1:]
        queries.append({"order_id": order_id, "cart_skus": cart, "target_sku": target})
    return queries


def _candidate_cross_entropy(recommendations, target_sku, epsilon=1e-12):
    """Calculate softmax loss within returned candidates, penalizing misses."""
    if recommendations.empty:
        return -log(epsilon)
    matches = recommendations.index[recommendations["recommended_sku"] == target_sku].tolist()
    if not matches:
        return -log(epsilon)
    target_position = recommendations.index.get_loc(matches[0])
    return softmax_cross_entropy(recommendations["recommendation_score"], target_position)


def evaluate_cart_engine(engine, orders, train_orders, k=10, candidate_k=100,
                         max_queries=250, random_state=42, queries=None):
    """Evaluate hidden basket products and return summary plus query outcomes."""
    graph = getattr(engine, "graph", None)
    if graph is None:
        graph = getattr(getattr(engine, "product2vec_model", None), "graph", None)
    known_skus = set(graph.nodes) if graph is not None else set()
    if queries is None:
        queries = leave_one_out_queries(orders, known_skus, max_queries, random_state)
    popularity = train_orders.groupby("SKU")["Order ID"].nunique().to_dict()
    catalog_recommendations = set()
    rows = []
    for query in queries:
        recommendations = engine.recommend(query["cart_skus"], top_n=candidate_k)
        ranked = recommendations["recommended_sku"].astype(str).tolist()
        top_k = ranked[:k]
        target = query["target_sku"]
        rank = ranked.index(target) + 1 if target in ranked else 0
        hit = int(target in top_k)
        catalog_recommendations.update(top_k)
        target_count = int(popularity.get(target, 0))
        rows.append({
            **query,
            "cart_skus": " | ".join(query["cart_skus"]),
            "cart_size": len(query["cart_skus"]),
            "target_train_orders": target_count,
            "popularity_segment": "rare" if target_count <= 2 else
                                  "medium" if target_count <= 10 else "popular",
            "basket_segment": "small" if len(query["cart_skus"]) == 1 else
                              "medium" if len(query["cart_skus"]) <= 3 else "large",
            "top_prediction": ranked[0] if ranked else pd.NA,
            "top_k_skus": " | ".join(top_k),
            "target_rank": rank,
            "hit_at_k": hit,
            "reciprocal_rank_at_k": 1.0 / rank if 0 < rank <= k else 0.0,
            "candidate_cross_entropy": _candidate_cross_entropy(recommendations, target),
        })
    outcomes = pd.DataFrame(rows)
    count = len(outcomes)
    summary = pd.DataFrame([{
        "k": k,
        "evaluated_orders": count,
        "precision_at_k": outcomes["hit_at_k"].sum() / (count * k) if count else 0.0,
        "recall_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "hit_rate_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "mrr_at_k": outcomes["reciprocal_rank_at_k"].mean() if count else 0.0,
        "mean_candidate_cross_entropy": outcomes["candidate_cross_entropy"].mean() if count else 0.0,
        "catalog_coverage_at_k": len(catalog_recommendations) / len(known_skus) if known_skus else 0.0,
    }])
    return summary, outcomes


def bootstrap_ranking_intervals(outcomes, n_bootstrap=2000, confidence=0.95,
                                random_state=42):
    """Bootstrap query-level confidence intervals for the major ranking metrics."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    required = {"hit_at_k", "reciprocal_rank_at_k"}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"Outcome columns are missing: {sorted(missing)}")
    if outcomes.empty:
        return pd.DataFrame(columns=["metric", "point_estimate", "lower_95",
                                     "upper_95", "bootstrap_samples"])
    values = {
        "hit_rate_at_k": outcomes["hit_at_k"].to_numpy(dtype=float),
        "recall_at_k": outcomes["hit_at_k"].to_numpy(dtype=float),
        "mrr_at_k": outcomes["reciprocal_rank_at_k"].to_numpy(dtype=float),
    }
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, len(outcomes), size=(n_bootstrap, len(outcomes)))
    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric, metric_values in values.items():
        estimates = metric_values[indices].mean(axis=1)
        rows.append({
            "metric": metric,
            "point_estimate": metric_values.mean(),
            "lower_95": np.quantile(estimates, alpha),
            "upper_95": np.quantile(estimates, 1.0 - alpha),
            "bootstrap_samples": n_bootstrap,
        })
    return pd.DataFrame(rows)


def segment_errors(outcomes):
    """Aggregate ranking quality by popularity and cart-size segments."""
    if outcomes.empty:
        return pd.DataFrame(columns=["segment_type", "segment", "queries", "hit_rate_at_k",
                                     "mrr_at_k", "mean_candidate_cross_entropy"])
    rows = []
    for segment_type, column in (("popularity", "popularity_segment"),
                                 ("basket_size", "basket_segment")):
        for segment, group in outcomes.groupby(column):
            rows.append({
                "segment_type": segment_type,
                "segment": segment,
                "queries": len(group),
                "hit_rate_at_k": group["hit_at_k"].mean(),
                "mrr_at_k": group["reciprocal_rank_at_k"].mean(),
                "mean_candidate_cross_entropy": group["candidate_cross_entropy"].mean(),
            })
    return pd.DataFrame(rows)


def _metadata_tokens(value):
    if pd.isna(value) or str(value).casefold() == "unknown":
        return set()
    return {token.strip().casefold() for token in re.split(r"[,;|/]", str(value)) if token.strip()}


def qualitative_samples(outcomes, product_catalog, sample_size=25, random_state=42):
    """Add target/prediction metadata and mismatch flags to sampled outcomes."""
    if outcomes.empty:
        return outcomes.copy()
    sample = outcomes.sample(min(sample_size, len(outcomes)), random_state=random_state).copy()
    fields = ["Product Name", "Κατηγορίες προϊόντων", "Προϊόν Ηλικία"]
    for role, sku_column in (("target", "target_sku"), ("predicted", "top_prediction")):
        for field in fields:
            mapping = product_catalog[field] if field in product_catalog else pd.Series(dtype=object)
            sample[f"{role}_{field}"] = sample[sku_column].map(mapping)
    target_age = sample["target_Προϊόν Ηλικία"].map(_metadata_tokens)
    predicted_age = sample["predicted_Προϊόν Ηλικία"].map(_metadata_tokens)
    target_category = sample["target_Κατηγορίες προϊόντων"].map(_metadata_tokens)
    predicted_category = sample["predicted_Κατηγορίες προϊόντων"].map(_metadata_tokens)
    sample["age_mismatch"] = [bool(left and right and left.isdisjoint(right))
                              for left, right in zip(target_age, predicted_age)]
    sample["category_mismatch"] = [bool(left and right and left.isdisjoint(right))
                                   for left, right in zip(target_category, predicted_category)]
    return sample
