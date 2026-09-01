"""Chronological splitting and offline ranking evaluation utilities."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


METRIC_NAMES = ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k")


def chronological_order_split(orders, train_fraction=0.70, dev_fraction=0.15):
    """Split complete orders chronologically into train, dev, and test frames."""
    required = {"Order ID", "Order Date", "SKU"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Required split columns are missing: {sorted(missing)}")
    if not 0 < train_fraction < 1 or not 0 < dev_fraction < 1:
        raise ValueError("train_fraction and dev_fraction must be between zero and one.")
    if train_fraction + dev_fraction >= 1:
        raise ValueError("train_fraction + dev_fraction must be less than one.")

    dated = orders.copy()
    dated["Order Date"] = pd.to_datetime(dated["Order Date"], errors="raise")
    order_dates = dated.groupby("Order ID")["Order Date"].min().reset_index()
    order_dates["_order_key"] = order_dates["Order ID"].astype(str)
    order_dates = order_dates.sort_values(["Order Date", "_order_key"])
    n_orders = len(order_dates)
    if n_orders < 3:
        raise ValueError("At least three orders are required for train/dev/test splitting.")

    train_end = max(1, min(int(n_orders * train_fraction), n_orders - 2))
    dev_end = max(train_end + 1, min(int(n_orders * (train_fraction + dev_fraction)), n_orders - 1))
    id_groups = (
        set(order_dates.iloc[:train_end]["Order ID"]),
        set(order_dates.iloc[train_end:dev_end]["Order ID"]),
        set(order_dates.iloc[dev_end:]["Order ID"]),
    )
    return tuple(dated[dated["Order ID"].isin(ids)].copy() for ids in id_groups)


def ranking_metrics(recommended, relevant, k):
    """Calculate Precision, Recall, Hit Rate, and reciprocal rank at K."""
    if k < 1:
        raise ValueError("k must be positive.")
    relevant = set(relevant)
    ranked = list(recommended)[:k]
    hits = [item in relevant for item in ranked]
    hit_count = sum(hits)
    reciprocal_rank = next((1.0 / (index + 1) for index, hit in enumerate(hits) if hit), 0.0)
    return {
        "precision_at_k": hit_count / k,
        "recall_at_k": hit_count / len(relevant) if relevant else 0.0,
        "hit_rate_at_k": float(hit_count > 0),
        "mrr_at_k": reciprocal_rank,
    }


def _evaluation_queries(orders, known_skus):
    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    queries = []
    skipped = 0
    for order_id, basket in baskets.items():
        known_basket = [sku for sku in basket if sku in known_skus]
        if len(known_basket) < 2:
            skipped += len(basket)
            continue
        for seed in known_basket:
            relevant = set(known_basket) - {seed}
            if relevant:
                queries.append((str(order_id), seed, relevant))
            else:
                skipped += 1
    return queries, skipped


def evaluate_recommender(model, orders, k=10, max_queries=None, random_state=42):
    """Evaluate one fitted SKU recommender on held-out order baskets."""
    known_skus = set(model.graph.nodes)
    queries, skipped = _evaluation_queries(orders, known_skus)
    total_eligible = len(queries)
    if max_queries is not None and len(queries) > max_queries:
        rng = np.random.default_rng(random_state)
        selected = np.sort(rng.choice(len(queries), size=max_queries, replace=False))
        queries = [queries[index] for index in selected]

    totals = defaultdict(float)
    recommended_catalog = set()
    cache = {}
    for _, seed, relevant in queries:
        if seed not in cache:
            frame = model.recommend(seed, top_n=k)
            cache[seed] = frame.get("recommended_sku", pd.Series(dtype=str)).astype(str).tolist()
        recommendations = cache[seed]
        recommended_catalog.update(recommendations)
        for name, value in ranking_metrics(recommendations, relevant, k).items():
            totals[name] += value

    evaluated = len(queries)
    metrics = {name: totals[name] / evaluated if evaluated else 0.0 for name in METRIC_NAMES}
    metrics.update({
        "catalog_coverage": len(recommended_catalog) / len(known_skus) if known_skus else 0.0,
        "evaluated_queries": evaluated,
        "eligible_queries": total_eligible,
        "skipped_queries": skipped,
    })
    return metrics


def single_sku_queries(orders, known_skus, max_queries=None, random_state=42):
    """Create one reproducible viewed-SKU/hidden-target query per order.

    Only products known to the fitted training catalog are eligible. Selecting
    one directed pair per order prevents large baskets from contributing more
    evaluation weight than small baskets.
    """
    known_skus = {str(sku) for sku in known_skus}
    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    eligible = [
        (str(order_id), [sku for sku in basket if sku in known_skus])
        for order_id, basket in baskets.items()
    ]
    eligible = [(order_id, basket) for order_id, basket in eligible if len(basket) >= 2]
    rng = np.random.default_rng(random_state)
    queries = []
    for order_id, basket in eligible:
        selected = rng.choice(len(basket), size=2, replace=False)
        queries.append({
            "order_id": order_id,
            "seed_sku": basket[int(selected[0])],
            "target_sku": basket[int(selected[1])],
            "held_out_basket_size": len(basket),
        })
    if max_queries is not None and len(queries) > max_queries:
        chosen = np.sort(rng.choice(len(queries), size=max_queries, replace=False))
        queries = [queries[index] for index in chosen]
    return queries


def evaluate_single_sku_engine(engine, queries, train_orders, catalog_skus,
                               k=10, candidate_k=100):
    """Evaluate product-page ranking from exactly one viewed SKU per query."""
    if k < 1:
        raise ValueError("k must be positive.")
    if candidate_k < k:
        raise ValueError("candidate_k must be greater than or equal to k.")
    catalog_skus = {str(sku) for sku in catalog_skus}
    popularity = train_orders.groupby("SKU")["Order ID"].nunique().to_dict()
    recommended_catalog = set()
    rows = []
    for query in queries:
        seed = str(query["seed_sku"])
        target = str(query["target_sku"])
        if hasattr(engine, "candidate_pool"):
            retrieved_pool = engine.candidate_pool([seed], top_n=candidate_k)
            recommendations = retrieved_pool.head(candidate_k)
        else:
            recommendations = engine.recommend([seed], top_n=candidate_k)
            retrieved_pool = recommendations
        ranked = recommendations.get(
            "recommended_sku", pd.Series(dtype=str)
        ).astype(str).tolist()
        retrieved = retrieved_pool.get(
            "recommended_sku", pd.Series(dtype=str)
        ).astype(str).tolist()
        top_k = ranked[:k]
        candidate_rank = ranked.index(target) + 1 if target in ranked else 0
        retrieved_rank = retrieved.index(target) + 1 if target in retrieved else 0
        target_sources = ""
        if retrieved_rank and "candidate_sources" in retrieved_pool:
            target_sources = str(
                retrieved_pool.iloc[retrieved_rank - 1]["candidate_sources"]
            )
        hit = int(target in top_k)
        target_count = int(popularity.get(target, 0))
        recommended_catalog.update(top_k)
        rows.append({
            **query,
            "seed_sku": seed,
            "target_sku": target,
            "target_train_orders": target_count,
            "popularity_segment": "rare" if target_count <= 2 else
                                  "medium" if target_count <= 10 else "popular",
            "top_prediction": top_k[0] if top_k else pd.NA,
            "top_k_skus": " | ".join(top_k),
            "recommendation_count": len(ranked),
            "retrieved_candidate_count": len(retrieved),
            "target_rank": candidate_rank,
            "target_retrieved_rank": retrieved_rank,
            "target_candidate_sources": target_sources,
            "hit_at_k": hit,
            "reciprocal_rank_at_k": (
                1.0 / candidate_rank if 0 < candidate_rank <= k else 0.0
            ),
            "target_in_candidates": int(candidate_rank > 0),
            "target_in_retrieved_pool": int(retrieved_rank > 0),
        })
    outcomes = pd.DataFrame(rows)
    count = len(outcomes)
    hit_sum = outcomes["hit_at_k"].sum() if count else 0
    summary = pd.DataFrame([{
        "k": k,
        "candidate_k": candidate_k,
        "evaluated_queries": count,
        "precision_at_k": hit_sum / (count * k) if count else 0.0,
        "recall_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "hit_rate_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "mrr_at_k": outcomes["reciprocal_rank_at_k"].mean() if count else 0.0,
        "candidate_recall": outcomes["target_in_candidates"].mean() if count else 0.0,
        "candidate_pool_recall": (
            outcomes["target_in_retrieved_pool"].mean() if count else 0.0
        ),
        "mean_retrieved_candidates": (
            outcomes["retrieved_candidate_count"].mean() if count else 0.0
        ),
        "catalog_coverage_at_k": (
            len(recommended_catalog) / len(catalog_skus) if catalog_skus else 0.0
        ),
    }])
    return summary, outcomes


def split_summary(train, dev, test):
    """Return order/line/basket/date statistics for each chronological split."""
    rows = []
    for name, frame in (("train", train), ("dev", dev), ("test", test)):
        dates = pd.to_datetime(frame["Order Date"])
        basket_sizes = frame.groupby("Order ID")["SKU"].nunique()
        rows.append({
            "split": name,
            "order_count": frame["Order ID"].nunique(),
            "line_count": len(frame),
            "unique_skus": frame["SKU"].nunique(),
            "multi_item_orders": int((basket_sizes >= 2).sum()),
            "start_date": dates.min(),
            "end_date": dates.max(),
        })
    return pd.DataFrame(rows)
