"""Evaluate the frozen recommender for a one-SKU product-page query."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_orders
from src.error_analysis import bootstrap_ranking_intervals
from src.evaluation import (chronological_order_split, evaluate_single_sku_engine,
                            single_sku_queries, split_summary)
from src.final_recommender import (HISTORICAL_TEST_CONFIG_PATH,
                                   FinalRecommender)
from src.metadata_recommender import metadata_tokens
from src.model_freeze import (assert_development_tuning_open,
                              assert_legacy_test_reuse_allowed)
from src.recommendation_engine import HybridRecommendationEngine


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_ROOT = Path("outputs/single_sku_evaluation")


class PopularityBaseline:
    """Rank training products globally or within the viewed product's category."""

    def __init__(self, orders, product_catalog, category_column=None):
        counts = orders.groupby("SKU")["Order ID"].nunique()
        self.counts = {str(sku): int(count) for sku, count in counts.items()}
        self.ranking = sorted(self.counts, key=lambda sku: (-self.counts[sku], sku))
        self.product_catalog = product_catalog
        self.category_column = category_column
        self.category_tokens = {}
        if category_column is not None:
            self.category_tokens = {
                str(sku): metadata_tokens(value)
                for sku, value in product_catalog[category_column].items()
            }

    def recommend(self, cart_skus, top_n=10, metadata_filters=None):
        """Return popular products, optionally prioritizing a shared category."""
        del metadata_filters
        query = list(dict.fromkeys(str(sku) for sku in cart_skus))
        query_set = set(query)
        candidates = [sku for sku in self.ranking if sku not in query_set]
        if self.category_column is not None:
            viewed_categories = set().union(
                *(self.category_tokens.get(sku, set()) for sku in query)
            )
            candidates.sort(key=lambda sku: (
                not bool(viewed_categories & self.category_tokens.get(sku, set())),
                -self.counts[sku],
                sku,
            ))
        maximum = max(self.counts.values(), default=1)
        rows = [{
            "cart_skus": " | ".join(query),
            "recommended_sku": sku,
            "recommendation_score": self.counts[sku] / maximum,
        } for sku in candidates[:top_n]]
        return pd.DataFrame(rows, columns=[
            "cart_skus", "recommended_sku", "recommendation_score"
        ])


def _component_engine(model, signal):
    """Create a single-signal view without fitting another model."""
    engine = model.engine
    return HybridRecommendationEngine(
        engine.copurchase_model,
        engine.product2vec_model,
        engine.adamic_adar_model,
        model.product_catalog,
        weights={signal: 1.0},
        metadata_model=engine.metadata_model,
        text_model=engine.text_model,
    )


def _hybrid_engine(model, weights):
    engine = model.engine
    return HybridRecommendationEngine(
        engine.copurchase_model,
        engine.product2vec_model,
        engine.adamic_adar_model,
        model.product_catalog,
        weights=weights,
        metadata_model=engine.metadata_model,
        text_model=engine.text_model,
    )


def four_signal_weight_grid(step=0.1):
    """Return coarse simplex weights with every production signal active."""
    units = round(1.0 / step)
    if step <= 0 or not abs(units * step - 1.0) < 1e-9 or units < 4:
        raise ValueError("step must divide one and leave room for four active signals.")
    signals = ("copurchase", "product2vec", "metadata", "text")
    rows = []
    for copurchase in range(1, units - 2):
        for product2vec in range(1, units - copurchase - 1):
            for metadata in range(1, units - copurchase - product2vec):
                text = units - copurchase - product2vec - metadata
                if text >= 1:
                    values = (copurchase, product2vec, metadata, text)
                    rows.append({
                        signal: value / units
                        for signal, value in zip(signals, values)
                    })
    return rows


def search_four_signal_weights(model, queries, training_orders, catalog_skus,
                               k=10, candidate_k=100, step=0.1):
    """Select a four-signal blend on Hit Rate, MRR, recall, then coverage."""
    rows = []
    for weights in four_signal_weight_grid(step):
        summary, _ = evaluate_single_sku_engine(
            _hybrid_engine(model, weights), queries, training_orders,
            catalog_skus, k, candidate_k,
        )
        row = {f"weight_{signal}": weight for signal, weight in weights.items()}
        row.update(summary.iloc[0].to_dict())
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        ["hit_rate_at_k", "mrr_at_k", "candidate_recall",
         "catalog_coverage_at_k"],
        ascending=False,
    ).reset_index(drop=True)
    return results


def _segment_metrics(outcomes):
    rows = []
    for (model, segment), group in outcomes.groupby(["model", "popularity_segment"]):
        rows.append({
            "model": model,
            "segment": segment,
            "queries": len(group),
            "hit_rate_at_k": group["hit_at_k"].mean(),
            "mrr_at_k": group["reciprocal_rank_at_k"].mean(),
            "candidate_recall": group["target_in_candidates"].mean(),
        })
    return pd.DataFrame(rows)


def run_evaluation(data_path=DATA_PATH, output_directory=None, split="development",
                   config_path=HISTORICAL_TEST_CONFIG_PATH, k=10, candidate_k=100,
                   max_queries=None, random_state=42, bootstrap_samples=2000,
                   tune_weights=False, weight_step=0.1):
    """Fit on earlier orders and compare product-page models on one-SKU queries."""
    if split not in {"development", "test"}:
        raise ValueError("split must be 'development' or 'test'.")
    if split == "test":
        assert_legacy_test_reuse_allowed()
    if tune_weights:
        if split != "development":
            raise ValueError("Weight tuning is restricted to the development split.")
        assert_development_tuning_open()
    output_directory = Path(output_directory or OUTPUT_ROOT / split)
    output_directory.mkdir(parents=True, exist_ok=True)

    orders = load_orders(data_path)
    train, development, test = chronological_order_split(orders)
    if split == "development":
        training = train
        evaluation = development
    else:
        training = pd.concat([train, development], ignore_index=True)
        evaluation = test

    model = FinalRecommender.from_config_path(config_path).fit(training)
    allowed_statuses = model.configuration.get("allowed_statuses")
    baseline_training = training if allowed_statuses is None else training[
        training["Order Status"].isin(allowed_statuses)
    ]
    catalog_skus = set(model.product_catalog.index.astype(str))
    queries = single_sku_queries(
        evaluation, catalog_skus, max_queries=max_queries, random_state=random_state
    )
    category_column = next(iter(model.configuration["metadata_field_weights"]))
    engines = {
        "global_popularity": PopularityBaseline(
            baseline_training, model.product_catalog
        ),
        "category_popularity": PopularityBaseline(
            baseline_training, model.product_catalog, category_column
        ),
        "copurchase": _component_engine(model, "copurchase"),
        "product2vec": _component_engine(model, "product2vec"),
        "metadata": _component_engine(model, "metadata"),
        "tfidf_name": _component_engine(model, "text"),
        "frozen_hybrid": model.engine,
    }
    weight_search = None
    selected_weights = None
    if tune_weights:
        weight_search = search_four_signal_weights(
            model, queries, baseline_training, catalog_skus, k, candidate_k,
            weight_step,
        )
        best = weight_search.iloc[0]
        selected_weights = {
            signal: float(best[f"weight_{signal}"])
            for signal in ("copurchase", "product2vec", "metadata", "text")
        }
        engines["tuned_four_signal"] = _hybrid_engine(model, selected_weights)

    summaries = []
    outcome_frames = []
    interval_frames = []
    for name, engine in engines.items():
        summary, outcomes = evaluate_single_sku_engine(
            engine, queries, baseline_training, catalog_skus, k, candidate_k
        )
        summary.insert(0, "model", name)
        summary.insert(1, "split", split)
        outcomes.insert(0, "model", name)
        intervals = bootstrap_ranking_intervals(
            outcomes, n_bootstrap=bootstrap_samples, random_state=random_state
        )
        intervals.insert(0, "model", name)
        summaries.append(summary)
        outcome_frames.append(outcomes)
        interval_frames.append(intervals)

    summary = pd.concat(summaries, ignore_index=True)
    outcomes = pd.concat(outcome_frames, ignore_index=True)
    intervals = pd.concat(interval_frames, ignore_index=True)
    segments = _segment_metrics(outcomes)
    query_frame = pd.DataFrame(queries)
    split_frame = split_summary(train, development, test)
    split_frame["split"] = split_frame["split"].replace({"dev": "development"})

    for frame, filename in (
        (summary, "summary_metrics.csv"),
        (outcomes, "prediction_outcomes.csv"),
        (intervals, "bootstrap_intervals.csv"),
        (segments, "popularity_segments.csv"),
        (query_frame, "queries.csv"),
        (split_frame, "split_summary.csv"),
    ):
        frame.to_csv(output_directory / filename, index=False, encoding="utf-8-sig")
    if weight_search is not None:
        weight_search.to_csv(
            output_directory / "four_signal_weight_search.csv",
            index=False, encoding="utf-8-sig",
        )
        selected_configuration = deepcopy(model.configuration)
        selected_configuration["selection_split"] = "development_single_sku"
        selected_configuration["blend_weights"] = selected_weights
        selected_configuration["tfidf"] = {
            "word_weight": model.engine.text_model.channel_weights["word"],
            "char_weight": model.engine.text_model.channel_weights["char"],
            "word_ngram_range": list(model.engine.text_model.word_ngram_range),
            "char_ngram_range": list(model.engine.text_model.char_ngram_range),
        }
        selected_row = summary[summary["model"] == "tuned_four_signal"].iloc[0]
        selected_configuration["development_single_sku_metrics"] = {
            metric: float(selected_row[metric]) for metric in (
                "hit_rate_at_k", "mrr_at_k", "candidate_recall",
                "catalog_coverage_at_k",
            )
        }
        (output_directory / "best_product_page_configuration.json").write_text(
            json.dumps(selected_configuration, indent=2), encoding="utf-8"
        )

    run_record = {
        "protocol": "one reproducibly sampled directed SKU pair per eligible order",
        "split": split,
        "training_orders": int(baseline_training["Order ID"].nunique()),
        "evaluation_orders": int(evaluation["Order ID"].nunique()),
        "eligible_queries": len(queries),
        "k": k,
        "candidate_k": candidate_k,
        "max_queries": max_queries,
        "random_state": random_state,
        "bootstrap_samples": bootstrap_samples,
        "config_path": str(config_path),
        "models": list(engines),
        "tuned_weights": selected_weights,
        "weight_step": weight_step if tune_weights else None,
    }
    (output_directory / "run_configuration.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print("Single-SKU product-page evaluation:\n")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {output_directory}")
    return summary, outcomes, intervals, segments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("development", "test"),
                        default="development")
    parser.add_argument("--config", type=Path, default=HISTORICAL_TEST_CONFIG_PATH)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--tune-weights", action="store_true")
    parser.add_argument("--weight-step", type=float, default=0.1)
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_evaluation(
        arguments.data, arguments.output, arguments.split, arguments.config,
        arguments.k, arguments.candidate_k, arguments.max_queries,
        arguments.seed, arguments.bootstrap_samples,
        arguments.tune_weights, arguments.weight_step,
    )


if __name__ == "__main__":
    main()
