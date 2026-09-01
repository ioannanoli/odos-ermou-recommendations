"""Compare the Phase 5 engine with metadata-bridged complement transfer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import create_product_catalog, load_orders
from src.error_analysis import evaluate_cart_engine, segment_errors
from src.evaluation import chronological_order_split, ranking_metrics
from src.metadata_transfer_recommender import MetadataEnhancedEngine, MetadataTransferRecommender
from src.model_config import SELECTED_PRODUCT2VEC_CONFIG
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import AdamicAdarRecommender, RecommendationEngine


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs/metadata_transfer")
TRANSFER_WEIGHTS = (0.10, 0.20, 0.30, 0.40)


def _fit_components(training_orders, random_state):
    catalog = create_product_catalog(training_orders)
    copurchase = CoPurchaseRecommender().fit(training_orders, catalog)
    product2vec = Product2VecRecommender(
        **SELECTED_PRODUCT2VEC_CONFIG, random_state=random_state
    ).fit(training_orders, catalog)
    adamic_adar = AdamicAdarRecommender().fit(graph=copurchase.graph)
    base_engine = RecommendationEngine(product2vec, adamic_adar, catalog)
    transfer = MetadataTransferRecommender().fit(
        product_catalog=catalog, graph=copurchase.graph
    )
    return base_engine, transfer


def _evaluate_named(model_name, engine, evaluation_orders, training_orders,
                    max_queries, random_state):
    summary, outcomes = evaluate_cart_engine(
        engine, evaluation_orders, training_orders,
        max_queries=max_queries, random_state=random_state,
    )
    summary.insert(0, "model", model_name)
    outcomes.insert(0, "model", model_name)
    segments = segment_errors(outcomes)
    segments.insert(0, "model", model_name)
    return summary, outcomes, segments


def _rare_hit_rate(segments):
    rare = segments[
        (segments["segment_type"] == "popularity") & (segments["segment"] == "rare")
    ]
    return float(rare.iloc[0]["hit_rate_at_k"]) if not rare.empty else 0.0


def evaluate_rare_seed_queries(engine, orders, training_orders, k=10,
                               max_queries=250, random_state=42):
    """Evaluate complements when a rare product is the observed query SKU."""
    known_skus = set(engine.product2vec_model.graph.nodes)
    popularity = training_orders.groupby("SKU")["Order ID"].nunique().to_dict()
    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: sorted(set(values.dropna().astype(str)))
    )
    queries = []
    for order_id, basket in baskets.items():
        known_basket = [sku for sku in basket if sku in known_skus]
        if len(known_basket) < 2:
            continue
        for seed in known_basket:
            if popularity.get(seed, 0) <= 2:
                queries.append((str(order_id), seed, set(known_basket) - {seed}))
    if max_queries is not None and len(queries) > max_queries:
        import numpy as np
        rng = np.random.default_rng(random_state)
        selected = np.sort(rng.choice(len(queries), size=max_queries, replace=False))
        queries = [queries[index] for index in selected]
    totals = defaultdict(float)
    for _, seed, relevant in queries:
        recommendations = engine.recommend([seed], top_n=k)
        ranked = recommendations.get("recommended_sku", pd.Series(dtype=str)).astype(str).tolist()
        for name, value in ranking_metrics(ranked, relevant, k).items():
            totals[name] += value
    count = len(queries)
    return {
        "rare_seed_queries": count,
        **{name: totals[name] / count if count else 0.0 for name in
           ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k")},
    }


def run_experiment(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY,
                   max_queries=250, random_state=42):
    """Evaluate base and metadata-enhanced engines on identical test queries."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    orders = load_orders(data_path)
    train, dev, test = chronological_order_split(orders)
    tuning_base, tuning_transfer = _fit_components(train, random_state)
    tuning_rows = []
    for weight in TRANSFER_WEIGHTS:
        engine = MetadataEnhancedEngine(tuning_base, tuning_transfer, transfer_weight=weight)
        summary, _, segments = _evaluate_named(
            f"metadata_{weight:.2f}", engine, dev, train, max_queries, random_state
        )
        rare_seed = evaluate_rare_seed_queries(
            engine, dev, train, max_queries=max_queries, random_state=random_state
        )
        tuning_rows.append({
            "transfer_weight": weight,
            "dev_hit_rate_at_k": summary.iloc[0]["hit_rate_at_k"],
            "dev_mrr_at_k": summary.iloc[0]["mrr_at_k"],
            "dev_rare_hit_rate_at_k": _rare_hit_rate(segments),
            "dev_rare_seed_hit_rate_at_k": rare_seed["hit_rate_at_k"],
            "dev_rare_seed_mrr_at_k": rare_seed["mrr_at_k"],
        })
    weight_search = pd.DataFrame(tuning_rows).sort_values(
        ["dev_rare_seed_hit_rate_at_k", "dev_rare_seed_mrr_at_k",
         "dev_hit_rate_at_k", "dev_mrr_at_k"],
        ascending=False,
    )
    selected_weight = float(weight_search.iloc[0]["transfer_weight"])

    train_dev = pd.concat([train, dev], ignore_index=True)
    base_engine, transfer = _fit_components(train_dev, random_state)
    enhanced_engine = MetadataEnhancedEngine(
        base_engine, transfer, transfer_weight=selected_weight
    )
    comparison = [
        _evaluate_named("base", base_engine, test, train_dev, max_queries, random_state),
        _evaluate_named("metadata_enhanced", enhanced_engine, test, train_dev,
                        max_queries, random_state),
    ]
    summary_comparison = pd.concat([item[0] for item in comparison], ignore_index=True)
    outcome_comparison = pd.concat([item[1] for item in comparison], ignore_index=True)
    segment_comparison = pd.concat([item[2] for item in comparison], ignore_index=True)
    summary_comparison.insert(1, "selected_transfer_weight", selected_weight)

    base_rows = outcome_comparison[outcome_comparison["model"] == "base"]
    enhanced_rows = outcome_comparison[outcome_comparison["model"] == "metadata_enhanced"]
    keys = ["order_id", "target_sku", "cart_skus"]
    changes = base_rows.merge(enhanced_rows, on=keys, suffixes=("_base", "_enhanced"))
    improvements = changes[
        (changes["hit_at_k_base"] == 0) & (changes["hit_at_k_enhanced"] == 1)
    ].copy()

    weight_search.to_csv(output_directory / "development_weight_search.csv", index=False,
                         encoding="utf-8-sig")
    summary_comparison.to_csv(output_directory / "summary_comparison.csv", index=False,
                              encoding="utf-8-sig")
    segment_comparison.to_csv(output_directory / "segment_comparison.csv", index=False,
                              encoding="utf-8-sig")
    outcome_comparison.to_csv(output_directory / "prediction_comparison.csv", index=False,
                              encoding="utf-8-sig")
    improvements.to_csv(output_directory / "improved_queries.csv", index=False,
                        encoding="utf-8-sig")
    rare_seed_comparison = pd.DataFrame([
        {"model": "base", **evaluate_rare_seed_queries(
            base_engine, test, train_dev, max_queries=max_queries, random_state=random_state
        )},
        {"model": "metadata_enhanced", **evaluate_rare_seed_queries(
            enhanced_engine, test, train_dev, max_queries=max_queries, random_state=random_state
        )},
    ])
    rare_seed_comparison.to_csv(output_directory / "rare_seed_comparison.csv", index=False,
                                encoding="utf-8-sig")
    print("Development weight search:\n", weight_search.to_string(index=False))
    print(f"\nSelected transfer weight: {selected_weight:.2f}")
    print("\nMetadata-transfer test comparison:\n", summary_comparison.to_string(index=False))
    rare = segment_comparison[
        (segment_comparison["segment_type"] == "popularity")
        & (segment_comparison["segment"] == "rare")
    ]
    print("\nRare-product comparison:\n", rare.to_string(index=False))
    print("\nRare product as query comparison:\n", rare_seed_comparison.to_string(index=False))
    return (summary_comparison, segment_comparison, outcome_comparison,
            improvements, rare_seed_comparison)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--max-queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_experiment(arguments.data, arguments.output, arguments.max_queries, arguments.seed)


if __name__ == "__main__":
    main()
