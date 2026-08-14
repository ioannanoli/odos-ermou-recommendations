"""Run Phase 6 quantitative evaluation and qualitative error analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import create_product_catalog, load_orders
from src.error_analysis import evaluate_cart_engine, qualitative_samples, segment_errors
from src.evaluation import chronological_order_split
from src.model_config import SELECTED_PRODUCT2VEC_CONFIG
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import AdamicAdarRecommender, RecommendationEngine


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs/phase6")


def run_analysis(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY,
                 max_queries=250, sample_size=25, random_state=42):
    """Fit on train+dev and analyze leave-one-out errors on untouched test orders."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    orders = load_orders(data_path)
    train, dev, test = chronological_order_split(orders)
    train_dev = pd.concat([train, dev], ignore_index=True)
    training_catalog = create_product_catalog(train_dev)
    analysis_catalog = create_product_catalog(orders)

    copurchase = CoPurchaseRecommender().fit(train_dev, training_catalog)
    product2vec = Product2VecRecommender(
        **SELECTED_PRODUCT2VEC_CONFIG, random_state=random_state
    ).fit(train_dev, training_catalog)
    adamic_adar = AdamicAdarRecommender().fit(graph=copurchase.graph)
    engine = RecommendationEngine(product2vec, adamic_adar, training_catalog)

    summary, outcomes = evaluate_cart_engine(
        engine, test, train_dev, max_queries=max_queries, random_state=random_state
    )
    segments = segment_errors(outcomes)
    samples = qualitative_samples(outcomes, analysis_catalog, sample_size, random_state)
    summary.to_csv(output_directory / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    outcomes.to_csv(output_directory / "prediction_outcomes.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(output_directory / "segment_errors.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(output_directory / "qualitative_samples.csv", index=False, encoding="utf-8-sig")
    print("Phase 6 summary:\n", summary.to_string(index=False))
    print("\nSegment analysis:\n", segments.to_string(index=False))
    return summary, outcomes, segments, samples


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--max-queries", type=int, default=250)
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_analysis(arguments.data, arguments.output, arguments.max_queries,
                 arguments.sample_size, arguments.seed)
