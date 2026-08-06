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
