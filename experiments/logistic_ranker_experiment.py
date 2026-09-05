"""Leakage-safe pre-freeze backtest of a learned logistic candidate ranker."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.expanded_candidate_generation import expanded_candidate_configuration
from serve_recommendations import DATA_PATH
from src.data_loader import load_orders
from src.error_analysis import bootstrap_ranking_intervals
from src.evaluation import single_sku_queries
from src.final_recommender import FinalRecommender, load_frozen_configuration
from src.logistic_ranker import (CandidateFeatureBuilder, FEATURE_NAMES,
                                 RegularizedLogisticRanker)


OUTPUT_DIRECTORY = Path("outputs/logistic_ranker_backtest")


def chronological_backtest_periods(orders, cutoff, fractions=(0.50, 0.20, 0.15, 0.15)):
    """Create four whole-order windows ending before frozen development."""
    if len(fractions) != 4 or any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must contain four positive values summing to one.")
    dated = orders.copy()
    dated["Order Date"] = pd.to_datetime(dated["Order Date"], errors="raise")
    dated = dated[dated["Order Date"] <= pd.Timestamp(cutoff)].copy()
    order_dates = dated.groupby("Order ID")["Order Date"].min().reset_index()
    order_dates["_order_key"] = order_dates["Order ID"].astype(str)
    order_dates = order_dates.sort_values(["Order Date", "_order_key"])
    if len(order_dates) < 8:
        raise ValueError("At least eight pre-freeze orders are required.")
    counts = [int(len(order_dates) * value) for value in fractions[:-1]]
    counts.append(len(order_dates) - sum(counts))
    boundaries = np.cumsum([0, *counts])
    periods = []
    for index in range(4):
        order_ids = set(order_dates.iloc[boundaries[index]:boundaries[index + 1]]["Order ID"])
        periods.append(dated[dated["Order ID"].isin(order_ids)].copy())
    return tuple(periods)


def _clear_caches(model):
    engine = model.engine
    for component in (
        engine.copurchase_model, engine.product2vec_model,
        engine.adamic_adar_model, engine.metadata_model, engine.text_model,
    ):
        cache = getattr(component, "_cart_cache", None)
        if cache is not None:
            cache.clear()


def _known_baskets(label_orders, known_skus):
    known_skus = {str(sku) for sku in known_skus}
    baskets = label_orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)) & known_skus)
    )
    return [(str(order_id), basket) for order_id, basket in baskets.items()
            if len(basket) >= 2]


def build_ranker_training_examples(model, history_orders, label_orders,
                                   negative_samples=30, progress_every=50):
    """Create positive basket pairs and hard retrieved negatives."""
    if negative_samples < 1:
        raise ValueError("negative_samples must be positive.")
    builder = CandidateFeatureBuilder(model, history_orders)
    baskets = _known_baskets(label_orders, model.product_catalog.index)
    feature_blocks = []
    label_blocks = []
    weight_blocks = []
    total_positive_pairs = 0
    retrieved_positive_pairs = 0
    seed_queries = 0
    for basket_position, (_, basket) in enumerate(baskets, 1):
        basket_set = set(basket)
        for seed in basket:
            seed_queries += 1
            positives = basket_set - {seed}
            total_positive_pairs += len(positives)
            pool = model.engine.candidate_pool([seed], top_n=10)
            candidate_skus = pool["recommended_sku"].astype(str).tolist()
            candidate_index = {sku: index for index, sku in enumerate(candidate_skus)}
            positive_indices = [candidate_index[sku] for sku in sorted(positives)
                                if sku in candidate_index]
            retrieved_positive_pairs += len(positive_indices)
            negative_indices = [
                index for index, candidate in enumerate(candidate_skus)
                if candidate not in basket_set
            ][:negative_samples]
            selected = positive_indices + negative_indices
            if positive_indices and negative_indices:
                all_features = builder.transform(seed, pool)
                labels = np.r_[
                    np.ones(len(positive_indices)), np.zeros(len(negative_indices))
                ]
                feature_blocks.append(all_features[selected])
                label_blocks.append(labels)
                # Every order contributes equal total mass regardless of basket size.
                row_weight = 1.0 / (len(basket) * len(selected))
                weight_blocks.append(np.full(len(selected), row_weight))
            _clear_caches(model)
        if progress_every and (basket_position % progress_every == 0 or
                               basket_position == len(baskets)):
            print(f"Prepared training baskets {basket_position:,}/{len(baskets):,}")
    if not feature_blocks:
        raise ValueError("No retrieved positive/negative training examples were available.")
    diagnostics = {
        "eligible_orders": len(baskets),
        "seed_queries": seed_queries,
        "directed_positive_pairs": total_positive_pairs,
        "retrieved_positive_pairs": retrieved_positive_pairs,
        "positive_pair_pool_recall": (
            retrieved_positive_pairs / total_positive_pairs
            if total_positive_pairs else 0.0
        ),
        "training_rows": int(sum(len(block) for block in label_blocks)),
        "positive_rows": int(sum(block.sum() for block in label_blocks)),
        "negative_rows": int(sum((block == 0).sum() for block in label_blocks)),
    }
    return (
        np.vstack(feature_blocks), np.concatenate(label_blocks),
        np.concatenate(weight_blocks), diagnostics,
    )


def prepare_evaluation_queries(model, history_orders, label_orders,
                               random_state=42, progress_every=25):
    """Prepare full candidate pools for one hidden target per later order."""
    queries = single_sku_queries(
        label_orders, model.product_catalog.index, random_state=random_state
    )
    builder = CandidateFeatureBuilder(model, history_orders)
    prepared = []
    for position, query in enumerate(queries, 1):
        seed = str(query["seed_sku"])
        pool = model.engine.candidate_pool([seed], top_n=10)
        prepared.append({
            **query,
            "seed_sku": seed,
            "target_sku": str(query["target_sku"]),
            "candidate_skus": pool["recommended_sku"].astype(str).tolist(),
            "fixed_scores": pool["recommendation_score"].to_numpy(dtype=float),
            "features": builder.transform(seed, pool),
        })
        _clear_caches(model)
        if progress_every and (position % progress_every == 0 or position == len(queries)):
            print(f"Prepared evaluation queries {position:,}/{len(queries):,}")
    return prepared


def evaluate_prepared_queries(prepared, catalog_skus, train_orders,
                              ranker=None, model_name="fixed_blend", k=10,
                              candidate_k=100):
    """Evaluate fixed ordering or logistic probabilities on shared pools."""
    if candidate_k < k:
        raise ValueError("candidate_k must be at least k.")
    popularity = train_orders.groupby("SKU")["Order ID"].nunique().to_dict()
    top_catalog = set()
    rows = []
    for query in prepared:
        candidates = query["candidate_skus"]
        if ranker is None:
            ranking_indices = np.arange(len(candidates))
        else:
            probabilities = ranker.predict_proba(query["features"])
            ranking_indices = np.lexsort((
                np.asarray(candidates, dtype=str),
                -query["fixed_scores"],
                -probabilities,
            ))
        ranked = [candidates[index] for index in ranking_indices]
        target = query["target_sku"]
        full_rank = ranked.index(target) + 1 if target in ranked else 0
        candidate_rank = full_rank if 0 < full_rank <= candidate_k else 0
        hit = int(0 < full_rank <= k)
        target_count = int(popularity.get(target, 0))
        top_catalog.update(ranked[:k])
        rows.append({
            "model": model_name,
            "order_id": query["order_id"],
            "seed_sku": query["seed_sku"],
            "target_sku": target,
            "held_out_basket_size": query["held_out_basket_size"],
            "target_train_orders": target_count,
            "popularity_segment": (
                "rare" if target_count <= 2 else
                "medium" if target_count <= 10 else "popular"
            ),
            "target_rank": int(full_rank),
            "hit_at_k": hit,
            "reciprocal_rank_at_k": 1.0 / full_rank if 0 < full_rank <= k else 0.0,
            "target_in_candidates": int(candidate_rank > 0),
            "target_in_retrieved_pool": int(full_rank > 0),
            "retrieved_candidate_count": len(ranked),
            "top_prediction": ranked[0] if ranked else pd.NA,
            "top_k_skus": " | ".join(ranked[:k]),
        })
    outcomes = pd.DataFrame(rows)
    count = len(outcomes)
    hit_sum = int(outcomes["hit_at_k"].sum()) if count else 0
    catalog_skus = {str(sku) for sku in catalog_skus}
    summary = pd.DataFrame([{
        "model": model_name,
        "k": k,
        "candidate_k": candidate_k,
        "evaluated_queries": count,
        "precision_at_k": hit_sum / (count * k) if count else 0.0,
        "recall_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "hit_rate_at_k": outcomes["hit_at_k"].mean() if count else 0.0,
        "mrr_at_k": outcomes["reciprocal_rank_at_k"].mean() if count else 0.0,
        "candidate_recall": outcomes["target_in_candidates"].mean() if count else 0.0,
        "candidate_pool_recall": outcomes["target_in_retrieved_pool"].mean() if count else 0.0,
        "catalog_coverage_at_k": len(top_catalog) / len(catalog_skus) if catalog_skus else 0.0,
    }])
    return summary, outcomes


def _period_summary(periods):
    names = ("foundation", "ranker_training", "ranker_validation", "backtest")
    rows = []
    for name, frame in zip(names, periods):
        dates = pd.to_datetime(frame["Order Date"])
        rows.append({
            "period": name,
            "orders": int(frame["Order ID"].nunique()),
            "lines": int(len(frame)),
            "unique_skus": int(frame["SKU"].nunique()),
            "start": str(dates.min()),
            "end": str(dates.max()),
        })
    return pd.DataFrame(rows)


def _segment_metrics(outcomes):
    return (outcomes.groupby(["model", "popularity_segment"])
            .agg(queries=("order_id", "size"),
                 hit_rate_at_10=("hit_at_k", "mean"),
                 mrr_at_10=("reciprocal_rank_at_k", "mean"),
                 candidate_recall_at_100=("target_in_candidates", "mean"),
                 candidate_pool_recall=("target_in_retrieved_pool", "mean"))
            .reset_index())


def run_logistic_backtest(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY,
                          negative_samples=30, l2_values=(0.01, 0.1, 1.0, 10.0),
                          k=10, candidate_k=100, random_state=42,
                          bootstrap_samples=2000):
    """Train, validate, and backtest logistic ranking before frozen development."""
    frozen_configuration = load_frozen_configuration()
    cutoff = pd.Timestamp(frozen_configuration["experiment_record"]["training_end"])
    orders = load_orders(data_path)
    periods = chronological_backtest_periods(orders, cutoff)
    foundation, ranker_training, ranker_validation, backtest = periods
    print("Pre-freeze periods:\n")
    print(_period_summary(periods).to_string(index=False))

    print("\nFitting snapshot 1 candidate models...")
    snapshot_one = FinalRecommender(expanded_candidate_configuration()).fit(foundation)
    train_x, train_y, train_weight, training_diagnostics = build_ranker_training_examples(
        snapshot_one, foundation, ranker_training, negative_samples
    )
    del snapshot_one
    gc.collect()

    history_two = pd.concat([foundation, ranker_training], ignore_index=True)
    print("\nFitting snapshot 2 candidate models...")
    snapshot_two = FinalRecommender(expanded_candidate_configuration()).fit(history_two)
    validation_prepared = prepare_evaluation_queries(
        snapshot_two, history_two, ranker_validation, random_state
    )
    search_rows = []
    for l2 in l2_values:
        ranker = RegularizedLogisticRanker(l2=l2).fit(
            train_x, train_y, train_weight
        )
        summary, _ = evaluate_prepared_queries(
            validation_prepared, snapshot_two.product_catalog.index, history_two,
            ranker, f"logistic_l2_{l2:g}", k, candidate_k,
        )
        search_rows.append({"l2": float(l2), **summary.iloc[0].to_dict()})
    regularization_search = pd.DataFrame(search_rows).sort_values(
        ["hit_rate_at_k", "mrr_at_k", "candidate_recall",
         "catalog_coverage_at_k", "l2"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    selected_l2 = float(regularization_search.iloc[0]["l2"])

    validation_x, validation_y, validation_weight, validation_diagnostics = (
        build_ranker_training_examples(
            snapshot_two, history_two, ranker_validation, negative_samples
        )
    )
    final_ranker = RegularizedLogisticRanker(l2=selected_l2).fit(
        np.vstack([train_x, validation_x]),
        np.concatenate([train_y, validation_y]),
        np.concatenate([train_weight, validation_weight]),
    )
    del validation_prepared, snapshot_two
    gc.collect()

    history_three = pd.concat(
        [foundation, ranker_training, ranker_validation], ignore_index=True
    )
    print("\nFitting snapshot 3 candidate models for untouched backtest...")
    snapshot_three = FinalRecommender(expanded_candidate_configuration()).fit(history_three)
    backtest_prepared = prepare_evaluation_queries(
        snapshot_three, history_three, backtest, random_state
    )
    fixed_summary, fixed_outcomes = evaluate_prepared_queries(
        backtest_prepared, snapshot_three.product_catalog.index, history_three,
        None, "fixed_40_10_10_40", k, candidate_k,
    )
    logistic_summary, logistic_outcomes = evaluate_prepared_queries(
        backtest_prepared, snapshot_three.product_catalog.index, history_three,
        final_ranker, "logistic_ranker", k, candidate_k,
    )
    summary = pd.concat([fixed_summary, logistic_summary], ignore_index=True)
    outcomes = pd.concat([fixed_outcomes, logistic_outcomes], ignore_index=True)
    segments = _segment_metrics(outcomes)
    intervals = []
    for model_name, group in outcomes.groupby("model"):
        frame = bootstrap_ranking_intervals(
            group, n_bootstrap=bootstrap_samples, random_state=random_state
        )
        frame.insert(0, "model", model_name)
        intervals.append(frame)
    intervals = pd.concat(intervals, ignore_index=True)
    paired = fixed_outcomes[["order_id", "hit_at_k", "target_rank"]].merge(
        logistic_outcomes[["order_id", "hit_at_k", "target_rank"]],
        on="order_id", suffixes=("_fixed", "_logistic"), validate="one_to_one",
    )
    paired_summary = {
        "logistic_only_hits": int(((paired["hit_at_k_logistic"] == 1) &
                                    (paired["hit_at_k_fixed"] == 0)).sum()),
        "fixed_only_hits": int(((paired["hit_at_k_fixed"] == 1) &
                                 (paired["hit_at_k_logistic"] == 0)).sum()),
        "both_hit": int(((paired["hit_at_k_fixed"] == 1) &
                          (paired["hit_at_k_logistic"] == 1)).sum()),
        "neither_hit": int(((paired["hit_at_k_fixed"] == 0) &
                             (paired["hit_at_k_logistic"] == 0)).sum()),
    }

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts = (
        (_period_summary(periods), "period_summary.csv"),
        (regularization_search, "regularization_search.csv"),
        (summary, "backtest_summary.csv"),
        (outcomes, "backtest_outcomes.csv"),
        (segments, "popularity_segments.csv"),
        (intervals, "bootstrap_intervals.csv"),
        (final_ranker.coefficient_frame(), "logistic_coefficients.csv"),
        (paired, "paired_outcomes.csv"),
    )
    for frame, filename in artifacts:
        frame.to_csv(output_directory / filename, index=False, encoding="utf-8-sig")
    run_record = {
        "protocol": "three-snapshot chronological logistic-ranking backtest",
        "frozen_development_queries_used": False,
        "latest_backtest_date": str(pd.to_datetime(backtest["Order Date"]).max()),
        "frozen_development_starts_after": str(cutoff),
        "negative_samples_per_seed": negative_samples,
        "feature_count": len(FEATURE_NAMES),
        "features": list(FEATURE_NAMES),
        "l2_values": [float(value) for value in l2_values],
        "selected_l2": selected_l2,
        "training_diagnostics": training_diagnostics,
        "validation_training_diagnostics": validation_diagnostics,
        "paired_backtest": paired_summary,
        "k": k,
        "candidate_k": candidate_k,
        "random_state": random_state,
        "bootstrap_samples": bootstrap_samples,
        "final_confirmation_required_on_new_orders": True,
    }
    (output_directory / "run_configuration.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print("\nRegularization search:\n")
    print(regularization_search.to_string(index=False))
    print("\nUntouched pre-freeze backtest:\n")
    print(summary.to_string(index=False))
    print(f"\nPaired outcomes: {paired_summary}")
    print(f"\nArtifacts written to {output_directory}")
    return summary, outcomes, regularization_search, final_ranker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--negative-samples", type=int, default=30)
    parser.add_argument("--l2-values", nargs="+", type=float,
                        default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_logistic_backtest(
        arguments.data, arguments.output, arguments.negative_samples,
        tuple(arguments.l2_values), arguments.k, arguments.candidate_k,
        arguments.seed, arguments.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
