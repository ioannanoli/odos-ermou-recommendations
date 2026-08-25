"""Evaluate the frozen product-page model once on genuinely later orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_orders
from src.error_analysis import bootstrap_ranking_intervals
from src.evaluation import evaluate_single_sku_engine, single_sku_queries
from src.final_recommender import FinalRecommender
from src.model_freeze import (DEFAULT_CONFIG_PATH, FREEZE_MANIFEST_PATH,
                              sha256_file, verify_frozen_artifacts,
                              verify_frozen_configuration)


HISTORY_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs/future_evaluation")


def strictly_future_orders(orders, cutoff):
    """Return order lines strictly later than the frozen serving cutoff."""
    required = {"Order ID", "Order Date", "SKU"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Future orders are missing columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff)
    dated = orders.copy()
    dated["Order Date"] = pd.to_datetime(
        dated["Order Date"], errors="raise", format="mixed"
    )
    return dated[dated["Order Date"] > cutoff].copy()


def _ensure_empty_output(output_directory):
    """Protect a future evaluation from being silently overwritten or repeated."""
    output_directory = Path(output_directory)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"Future evaluation output already exists: {output_directory}. "
            "Use a new directory for a genuinely new evaluation period."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def run_future_evaluation(future_path, history_path=HISTORY_PATH,
                          output_directory=OUTPUT_DIRECTORY, k=10,
                          candidate_k=100, random_state=42,
                          bootstrap_samples=2000):
    """Fit the locked model on frozen history and evaluate all eligible later orders."""
    manifest = verify_frozen_configuration(
        DEFAULT_CONFIG_PATH, FREEZE_MANIFEST_PATH
    )
    verify_frozen_artifacts(FREEZE_MANIFEST_PATH)
    cutoff = pd.Timestamp(manifest["serving_training_cutoff"])
    history = load_orders(history_path)
    history_dates = pd.to_datetime(history["Order Date"], errors="raise")
    if history_dates.max() > cutoff:
        raise ValueError(
            "The history workbook contains orders after the frozen training cutoff. "
            "Do not expand training data before the locked future evaluation."
        )

    future_export = load_orders(future_path)
    future = strictly_future_orders(future_export, cutoff)
    if future.empty:
        raise ValueError(
            "No orders occur strictly after the frozen serving cutoff "
            f"({cutoff}). Wait for a genuinely newer export."
        )
    overlapping_orders = set(history["Order ID"]).intersection(future["Order ID"])
    if overlapping_orders:
        raise ValueError(
            "Future evaluation order IDs overlap frozen training history; "
            "the periods are not independent."
        )

    model = FinalRecommender.from_config_path(DEFAULT_CONFIG_PATH).fit(history)
    catalog_skus = set(model.product_catalog.index.astype(str))
    queries = single_sku_queries(
        future, catalog_skus, max_queries=None, random_state=random_state
    )
    if not queries:
        raise ValueError(
            "The later period contains no orders with two products known to the "
            "frozen catalog, so HR@K cannot be evaluated yet."
        )

    summary, outcomes = evaluate_single_sku_engine(
        model.engine, queries, history, catalog_skus, k, candidate_k
    )
    summary.insert(0, "model", "frozen_tfidf_hybrid")
    summary.insert(1, "split", "strictly_future")
    intervals = bootstrap_ranking_intervals(
        outcomes, n_bootstrap=bootstrap_samples, random_state=random_state
    )
    segments = outcomes.groupby("popularity_segment").agg(
        queries=("order_id", "size"),
        hit_rate_at_k=("hit_at_k", "mean"),
        mrr_at_k=("reciprocal_rank_at_k", "mean"),
        candidate_recall=("target_in_candidates", "mean"),
    ).reset_index().rename(columns={"popularity_segment": "segment"})

    output_directory = _ensure_empty_output(output_directory)
    summary.to_csv(output_directory / "summary_metrics.csv", index=False,
                   encoding="utf-8-sig")
    outcomes.to_csv(output_directory / "prediction_outcomes.csv", index=False,
                    encoding="utf-8-sig")
    intervals.to_csv(output_directory / "bootstrap_intervals.csv", index=False,
                     encoding="utf-8-sig")
    segments.to_csv(output_directory / "popularity_segments.csv", index=False,
                    encoding="utf-8-sig")
    pd.DataFrame(queries).to_csv(output_directory / "queries.csv", index=False,
                                 encoding="utf-8-sig")
    run_record = {
        "protocol": "locked single-SKU future-period evaluation; no tuning",
        "configuration_sha256": manifest["configuration_sha256"],
        "freeze_manifest": str(FREEZE_MANIFEST_PATH),
        "history_path": str(history_path),
        "history_sha256": sha256_file(history_path),
        "future_path": str(future_path),
        "future_sha256": sha256_file(future_path),
        "training_cutoff": str(cutoff),
        "future_period_start": str(pd.to_datetime(future["Order Date"]).min()),
        "future_period_end": str(pd.to_datetime(future["Order Date"]).max()),
        "future_orders": int(future["Order ID"].nunique()),
        "eligible_queries": len(queries),
        "k": k,
        "candidate_k": candidate_k,
        "random_state": random_state,
        "bootstrap_samples": bootstrap_samples,
        "tuning_performed": False,
    }
    (output_directory / "run_configuration.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print("Frozen future-period evaluation:\n")
    print(summary.to_string(index=False))
    print(f"\nArtifacts written to {output_directory}")
    return summary, outcomes, intervals, segments


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future", type=Path, required=True,
                        help="A new order export containing dates after the cutoff.")
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_future_evaluation(
        arguments.future, arguments.history, arguments.output, arguments.k,
        arguments.candidate_k, arguments.seed, arguments.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
