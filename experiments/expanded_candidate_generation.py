"""Train and inspect the experimental Version 2 expanded candidate pool."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

import pandas as pd

from serve_recommendations import DATA_PATH
from src.data_loader import load_orders
from src.final_recommender import FinalRecommender, load_frozen_configuration
from src.model_freeze import DEFAULT_CONFIG_PATH


MODEL_PATH = Path("models/experimental_v2_expanded_candidates.pkl")
DEFAULT_CANDIDATE_GENERATION = {
    "minimum_per_source": 500,
    "multiplier": 30,
    "maximum_per_source": 3000,
    "source_limits": {},
    "track_sources": True,
}


def expanded_candidate_configuration(base_config_path=DEFAULT_CONFIG_PATH):
    """Derive an unfrozen V2 experiment without modifying frozen Version 1."""
    configuration = deepcopy(load_frozen_configuration(base_config_path))
    for key in (
        "selection_split", "development_queries", "development_metrics",
        "development_single_sku_metrics", "experiment_record",
    ):
        configuration.pop(key, None)
    configuration["model_version"] = "v2_expanded_candidates_experimental"
    configuration["candidate_generation"] = deepcopy(
        DEFAULT_CANDIDATE_GENERATION
    )
    return configuration


def train_experimental_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Fit and save V2 candidate generation while preserving the V1 artifact."""
    model = FinalRecommender(expanded_candidate_configuration())
    model.fit(load_orders(data_path))
    model.save(model_path)
    return model


def candidate_pool_diagnostics(model, sku, ranking_top_n=10):
    """Return the full pool, per-source counts, and its final top ranking."""
    pool = model.engine.candidate_pool([sku], top_n=ranking_top_n)
    if pool.empty:
        counts = pd.DataFrame(columns=["source", "candidates"])
    else:
        source_counts = Counter(
            source
            for value in pool["candidate_sources"]
            for source in str(value).split(" | ")
            if source
        )
        counts = pd.DataFrame(
            source_counts.items(), columns=["source", "candidates"]
        ).sort_values(["candidates", "source"], ascending=[False, True])
    return pool, counts, pool.head(ranking_top_n).reset_index(drop=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train the separate V2 model.")
    train.add_argument("--data", type=Path, default=DATA_PATH)
    train.add_argument("--model", type=Path, default=MODEL_PATH)

    inspect = subparsers.add_parser(
        "inspect", help="Inspect the retrieved pool for one product SKU."
    )
    inspect.add_argument("sku")
    inspect.add_argument("--model", type=Path, default=MODEL_PATH)
    inspect.add_argument("--top-n", type=int, default=10)
    inspect.add_argument("--output", type=Path)
    return parser


def main():
    sys.stdout.reconfigure(errors="backslashreplace")
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        model = train_experimental_model(arguments.data, arguments.model)
        print(f"Saved experimental V2 model to {arguments.model}")
        print(json.dumps(model.training_summary, ensure_ascii=True, indent=2))
        return

    model = FinalRecommender.load(arguments.model)
    pool, counts, ranking = candidate_pool_diagnostics(
        model, arguments.sku, arguments.top_n
    )
    print(f"Retrieved {len(pool)} unique candidates for {arguments.sku}.\n")
    print("Candidate sources:")
    print(counts.to_string(index=False))
    print("\nFinal ranking:")
    print(ranking.to_string(index=False))
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(arguments.output, index=False, encoding="utf-8-sig")
        print(f"\nSaved the full candidate pool to {arguments.output}")


if __name__ == "__main__":
    main()
