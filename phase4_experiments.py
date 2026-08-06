"""Run chronological evaluation and random Product2Vec hyperparameter search."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import load_orders
from src.evaluation import chronological_order_split, evaluate_recommender, split_summary
from src.product2vec_recommender import Product2VecRecommender


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs/phase4")
K = 10

SEARCH_SPACE = {
    "n_components": [16, 32, 48],
    "walk_length": [6, 8, 10],
    "walks_per_node": [1, 2],
    "window_size": [2, 3, 4],
    "negative_samples": [2, 3],
    "epochs": [1, 2],
    "learning_rate": [0.01, 0.025, 0.05],
    "p": [0.5, 1.0, 2.0],
    "q": [0.5, 1.0, 2.0],
}


def sample_configurations(n_trials, random_state=42):
    """Draw unique, reproducible configurations from the discrete search space."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    rng = np.random.default_rng(random_state)
    configurations, seen = [], set()
    while len(configurations) < n_trials:
        config = {name: values[rng.integers(len(values))] for name, values in SEARCH_SPACE.items()}
        signature = tuple(config.items())
        if signature not in seen:
            seen.add(signature)
            configurations.append(config)
    return configurations


def _prefixed(metrics, prefix):
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _diagnosis(train_hit_rate, dev_hit_rate):
    """Give a simple train/dev diagnostic without changing model selection."""
    gap = train_hit_rate - dev_hit_rate
    if train_hit_rate < 0.10:
        return "possible_high_bias"
    if gap > 0.10:
        return "possible_high_variance"
    return "no_large_gap_detected"


def run_experiments(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY,
                    n_trials=4, max_queries=250, random_state=42):
    """Tune on dev, refit the winner on train+dev, and evaluate test once."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    orders = load_orders(data_path)
    train, dev, test = chronological_order_split(orders)
    split_summary(train, dev, test).to_csv(
        output_directory / "split_summary.csv", index=False, encoding="utf-8-sig"
    )

    baseline = CoPurchaseRecommender().fit(train)
    baseline_rows = []
    for split_name, frame in (("train", train), ("dev", dev)):
        metrics = evaluate_recommender(
            baseline, frame, k=K, max_queries=max_queries, random_state=random_state
        )
        baseline_rows.append({"model": "copurchase", "split": split_name, **metrics})
    pd.DataFrame(baseline_rows).to_csv(
        output_directory / "baseline_metrics.csv", index=False, encoding="utf-8-sig"
    )

    experiment_rows = []
    for trial, config in enumerate(sample_configurations(n_trials, random_state), start=1):
        print(f"Trial {trial}/{n_trials}: {config}")
        model = Product2VecRecommender(**config, random_state=random_state).fit(train)
        train_metrics = evaluate_recommender(
            model, train, k=K, max_queries=max_queries, random_state=random_state
        )
        dev_metrics = evaluate_recommender(
            model, dev, k=K, max_queries=max_queries, random_state=random_state
        )
        gap = train_metrics["hit_rate_at_k"] - dev_metrics["hit_rate_at_k"]
        experiment_rows.append({
            "trial": trial,
            **config,
            **_prefixed(train_metrics, "train"),
            **_prefixed(dev_metrics, "dev"),
            "hit_rate_generalization_gap": gap,
            "diagnosis": _diagnosis(train_metrics["hit_rate_at_k"], dev_metrics["hit_rate_at_k"]),
        })

    experiments = pd.DataFrame(experiment_rows).sort_values(
        ["dev_hit_rate_at_k", "dev_mrr_at_k", "dev_recall_at_k"], ascending=False
    )
    experiments.to_csv(
        output_directory / "hyperparameter_results.csv", index=False, encoding="utf-8-sig"
    )
    parameter_names = list(SEARCH_SPACE)
    best_config = experiments.iloc[0][parameter_names].to_dict()
    for name in ("n_components", "walk_length", "walks_per_node", "window_size",
                 "negative_samples", "epochs"):
        best_config[name] = int(best_config[name])

    train_dev = pd.concat([train, dev], ignore_index=True)
    final_models = {
        "copurchase": CoPurchaseRecommender().fit(train_dev),
        "product2vec": Product2VecRecommender(
            **best_config, random_state=random_state
        ).fit(train_dev),
    }
    test_rows = []
    for model_name, model in final_models.items():
        metrics = evaluate_recommender(
            model, test, k=K, max_queries=max_queries, random_state=random_state
        )
        test_rows.append({"model": model_name, "split": "test", **metrics})
    test_metrics = pd.DataFrame(test_rows)
    test_metrics.to_csv(output_directory / "test_metrics.csv", index=False, encoding="utf-8-sig")
    print("\nBest Product2Vec configuration:", best_config)
    print("\nFinal test metrics:\n", test_metrics.to_string(index=False))
    return experiments, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--max-queries", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_experiments(arguments.data, arguments.output, arguments.trials,
                    arguments.max_queries, arguments.seed)
