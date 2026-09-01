"""Audit Version 2 candidate retrieval without reusing evaluation targets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from experiments.expanded_candidate_generation import MODEL_PATH
from src.final_recommender import FinalRecommender
from src.metadata_recommender import metadata_tokens
from src.recommendation_engine import HybridRecommendationEngine


OUTPUT_DIRECTORY = Path("outputs/v2_candidate_health")
PRODUCT_NAME = "Product Name"
SOURCE_SIGNALS = tuple(HybridRecommendationEngine.SCORE_COLUMNS)


def _name_language(value):
    """Classify a product name by the scripts relevant to this catalog."""
    text = "" if pd.isna(value) else str(value)
    has_greek = any("\u0370" <= character <= "\u03ff" or
                    "\u1f00" <= character <= "\u1fff" for character in text)
    has_latin = any(("a" <= character.casefold() <= "z") for character in text)
    if has_greek and has_latin:
        return "mixed_greek_latin"
    if has_greek:
        return "greek_only"
    if has_latin:
        return "latin_only"
    return "missing_or_other"


def _clear_candidate_caches(engine):
    """Bound memory while auditing thousands of independent product queries."""
    models = (
        engine.copurchase_model, engine.product2vec_model,
        engine.adamic_adar_model, engine.metadata_model, engine.text_model,
    )
    for component in models:
        cache = getattr(component, "_cart_cache", None)
        if cache is not None:
            cache.clear()


def _candidate_sources(pool):
    """Return source sets for every candidate in one retrieved pool."""
    if "candidate_sources" not in pool:
        raise ValueError(
            "The model does not track candidate sources. Train the experimental "
            "Version 2 model before running this report."
        )
    return [
        {source for source in str(value).split(" | ") if source}
        for value in pool["candidate_sources"]
    ]


def _pool_is_deterministic(left, right):
    columns = ["recommended_sku", "recommendation_score"]
    if len(left) != len(right) or any(column not in left or column not in right
                                     for column in columns):
        return False
    return (
        left["recommended_sku"].astype(str).tolist()
        == right["recommended_sku"].astype(str).tolist()
        and np.allclose(
            left["recommendation_score"].to_numpy(dtype=float),
            right["recommendation_score"].to_numpy(dtype=float),
            rtol=0.0, atol=1e-12,
        )
    )


def audit_candidate_health(model, top_n=10, max_skus=None, random_state=42,
                           check_determinism=True, progress_every=250):
    """Audit unlabeled retrieval health across the fitted product catalog."""
    if model.engine is None:
        raise RuntimeError("Load or fit a model before running the health audit.")
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if not model.engine.candidate_generation.get("track_sources"):
        raise ValueError("Candidate source tracking must be enabled for this audit.")

    catalog = model.product_catalog
    skus = np.array(sorted(catalog.index.astype(str)), dtype=object)
    if max_skus is not None:
        if max_skus < 1:
            raise ValueError("max_skus must be positive when supplied.")
        if max_skus < len(skus):
            rng = np.random.default_rng(random_state)
            skus = np.sort(rng.choice(skus, size=max_skus, replace=False))

    graph = model.engine.copurchase_model.graph
    metadata_fields = tuple(model.engine.metadata_model.field_weights)
    weights = model.engine.weights
    detail_rows = []
    source_combinations = Counter()
    top_catalog = set()

    for position, raw_sku in enumerate(skus, 1):
        sku = str(raw_sku)
        started = perf_counter()
        error = ""
        try:
            pool = model.engine.candidate_pool([sku], top_n=top_n)
            latency_ms = (perf_counter() - started) * 1000.0
            sources = _candidate_sources(pool)
            if check_determinism:
                repeated = model.engine.candidate_pool([sku], top_n=top_n)
                deterministic = _pool_is_deterministic(pool, repeated)
            else:
                deterministic = True

            recommended = pool.get(
                "recommended_sku", pd.Series(dtype=str)
            ).astype(str)
            top_catalog.update(recommended.head(top_n))
            for source_set in sources:
                source_combinations[" | ".join(sorted(source_set))] += 1

            expected_score = sum(
                weights[signal] * pool[score_column]
                for signal, score_column in model.engine.SCORE_COLUMNS.items()
            ) if not pool.empty else pd.Series(dtype=float)
            formula_matches = bool(
                np.allclose(
                    pool["recommendation_score"].to_numpy(dtype=float),
                    expected_score.to_numpy(dtype=float),
                    rtol=0.0, atol=1e-12,
                )
            )
            score_columns = list(model.engine.SCORE_COLUMNS.values()) + [
                "recommendation_score"
            ]
            scores_bounded = all(
                pool[column].between(-1e-12, 1.0 + 1e-12).all()
                for column in score_columns
            )
            source_counts = {
                signal: sum(signal in source_set for source_set in sources)
                for signal in SOURCE_SIGNALS
            }
            invariant_ok = all((
                sku not in set(recommended),
                recommended.is_unique,
                pool["recommendation_score"].is_monotonic_decreasing,
                formula_matches,
                scores_bounded,
                deterministic,
            ))
        except Exception as exception:
            latency_ms = (perf_counter() - started) * 1000.0
            pool = pd.DataFrame()
            sources = []
            source_counts = {signal: 0 for signal in SOURCE_SIGNALS}
            deterministic = formula_matches = scores_bounded = invariant_ok = False
            error = f"{type(exception).__name__}: {exception}"

        graph_attributes = graph.nodes[sku] if sku in graph else {}
        order_count = int(graph_attributes.get("order_count", 0))
        graph_degree = int(graph.degree(sku)) if sku in graph else 0
        product_name = catalog.at[sku, PRODUCT_NAME] if PRODUCT_NAME in catalog else ""
        missing_metadata = sum(
            not metadata_tokens(catalog.at[sku, field])
            for field in metadata_fields
        )
        detail_rows.append({
            "sku": sku,
            "product_name": "" if pd.isna(product_name) else str(product_name),
            "name_language": _name_language(product_name),
            "training_order_count": order_count,
            "popularity_segment": (
                "rare" if order_count <= 2 else
                "medium" if order_count <= 10 else "popular"
            ),
            "graph_degree": graph_degree,
            "is_graph_isolated": int(graph_degree == 0),
            "missing_metadata_fields": int(missing_metadata),
            "candidate_pool_size": int(len(pool)),
            "top_n_result_count": int(min(len(pool), top_n)),
            "short_top_n": int(len(pool) < top_n),
            **{
                f"{signal}_candidate_count": int(source_counts[signal])
                for signal in SOURCE_SIGNALS
            },
            "multi_source_candidate_count": int(sum(len(value) > 1 for value in sources)),
            "mean_sources_per_candidate": (
                float(np.mean([len(value) for value in sources])) if sources else 0.0
            ),
            "latency_ms": float(latency_ms),
            "seed_excluded": int(pool.empty or sku not in set(
                pool.get("recommended_sku", pd.Series(dtype=str)).astype(str)
            )),
            "candidates_unique": int(
                pool.empty or pool["recommended_sku"].astype(str).is_unique
            ),
            "score_sorted": int(
                pool.empty or pool["recommendation_score"].is_monotonic_decreasing
            ),
            "formula_matches": int(formula_matches),
            "scores_bounded": int(scores_bounded),
            "deterministic": int(deterministic),
            "invariants_passed": int(invariant_ok),
            "error": error,
        })
        _clear_candidate_caches(model.engine)
        if progress_every and (position % progress_every == 0 or position == len(skus)):
            print(f"Audited {position:,}/{len(skus):,} catalog SKUs")

    details = pd.DataFrame(detail_rows)
    successful = details[details["error"] == ""]
    latency = successful["latency_ms"]
    pool_sizes = successful["candidate_pool_size"]
    summary = {
        "protocol": "unlabeled all-catalog candidate health audit",
        "evaluated_skus": int(len(details)),
        "catalog_skus": int(len(catalog)),
        "sampling_used": bool(len(details) < len(catalog)),
        "top_n": int(top_n),
        "failures": int((details["error"] != "").sum()),
        "invariant_failures": int((details["invariants_passed"] == 0).sum()),
        "zero_candidate_skus": int((details["candidate_pool_size"] == 0).sum()),
        "short_top_n_skus": int(details["short_top_n"].sum()),
        "unique_top_n_catalog_skus": int(len(top_catalog)),
        "top_n_catalog_coverage": float(len(top_catalog) / len(catalog)) if len(catalog) else 0.0,
        "mean_candidate_pool_size": float(pool_sizes.mean()) if len(pool_sizes) else 0.0,
        "median_candidate_pool_size": float(pool_sizes.median()) if len(pool_sizes) else 0.0,
        "p95_candidate_pool_size": float(pool_sizes.quantile(0.95)) if len(pool_sizes) else 0.0,
        "mean_latency_ms": float(latency.mean()) if len(latency) else 0.0,
        "median_latency_ms": float(latency.median()) if len(latency) else 0.0,
        "p95_latency_ms": float(latency.quantile(0.95)) if len(latency) else 0.0,
        "maximum_latency_ms": float(latency.max()) if len(latency) else 0.0,
        "accuracy_metrics_calculated": False,
    }

    segment_rows = []
    source_rows = []
    for segment, group in details.groupby("popularity_segment", sort=False):
        segment_rows.append({
            "segment": segment,
            "skus": int(len(group)),
            "zero_candidate_rate": float((group["candidate_pool_size"] == 0).mean()),
            "short_top_n_rate": float(group["short_top_n"].mean()),
            "mean_candidate_pool_size": float(group["candidate_pool_size"].mean()),
            "median_candidate_pool_size": float(group["candidate_pool_size"].median()),
            "p95_latency_ms": float(group["latency_ms"].quantile(0.95)),
            "invariant_failures": int((group["invariants_passed"] == 0).sum()),
        })
    for signal in SOURCE_SIGNALS:
        counts = details[f"{signal}_candidate_count"]
        source_rows.append({
            "source": signal,
            "candidate_occurrences": int(counts.sum()),
            "skus_with_candidates": int((counts > 0).sum()),
            "mean_candidates_per_sku": float(counts.mean()),
        })
    combinations = pd.DataFrame([
        {"source_combination": combination, "candidate_occurrences": count}
        for combination, count in source_combinations.most_common()
    ])
    return (
        summary,
        details,
        pd.DataFrame(segment_rows),
        pd.DataFrame(source_rows),
        combinations,
    )


def run_candidate_health_report(model_path=MODEL_PATH,
                                output_directory=OUTPUT_DIRECTORY, top_n=10,
                                max_skus=None, random_state=42,
                                check_determinism=True):
    """Load Version 2, run the audit, and write reproducible CSV/JSON outputs."""
    model_path = Path(model_path)
    model = FinalRecommender.load(model_path)
    result = audit_candidate_health(
        model, top_n=top_n, max_skus=max_skus, random_state=random_state,
        check_determinism=check_determinism,
    )
    summary, details, segments, sources, combinations = result
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    for frame, filename in (
        (details, "product_health.csv"),
        (segments, "popularity_health.csv"),
        (sources, "source_contributions.csv"),
        (combinations, "source_combinations.csv"),
    ):
        frame.to_csv(output_directory / filename, index=False, encoding="utf-8-sig")
    run_record = {
        **summary,
        "model_path": str(model_path),
        "model_version": model.configuration.get("model_version"),
        "candidate_generation": model.engine.candidate_generation,
        "blend_weights": model.engine.weights,
        "random_state": int(random_state),
        "determinism_checked": bool(check_determinism),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(run_record, indent=2), encoding="utf-8"
    )
    print("\nCandidate health summary:")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts written to {output_directory}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--max-skus", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-determinism", action="store_true")
    return parser.parse_args()


def main():
    arguments = parse_args()
    run_candidate_health_report(
        arguments.model, arguments.output, arguments.top_n,
        arguments.max_skus, arguments.seed, not arguments.skip_determinism,
    )


if __name__ == "__main__":
    main()
