"""Leakage-safe development search and one-time final test evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import create_product_catalog, load_orders
from src.error_analysis import (bootstrap_ranking_intervals, evaluate_cart_engine,
                                leave_one_out_queries, segment_errors)
from src.evaluation import chronological_order_split, split_summary
from src.heterogeneous_graph import HeterogeneousProduct2VecRecommender
from src.metadata_recommender import (DEFAULT_METADATA_WEIGHTS,
                                      MetadataSimilarityRecommender)
from src.model_config import SELECTED_PRODUCT2VEC_CONFIG
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import (AdamicAdarRecommender,
                                       HybridRecommendationEngine)


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_ROOT = Path("outputs/improvement_experiments")
RANDOM_STATE = 42
K = 10
HALF_LIVES = (None, 6, 12, 18, 24, 36)
P2V_SEARCH_SPACE = {
    "n_components": (32, 48, 64, 96),
    "walk_length": (6, 8, 12, 16),
    "walks_per_node": (2, 4, 6, 8),
    "window_size": (2, 3, 4, 5),
    "negative_samples": (2, 4, 6),
    "epochs": (2, 3, 5),
    "learning_rate": (0.01, 0.025, 0.05),
    "p": (0.5, 1.0, 2.0),
    "q": (0.5, 1.0, 2.0),
}


def inspect_status_policies(orders):
    """Derive defensible policies from statuses actually present in the export."""
    observed = set(orders["Order Status"].dropna().astype(str))
    completed = {status for status in observed if status == "wc-completed"}
    successful = observed.intersection({"wc-completed", "wc-pickup", "wc-ready-pickup"})
    intent = observed.intersection({
        "wc-completed", "wc-pickup", "wc-ready-pickup", "wc-processing",
        "wc-on-hold", "wc-pending",
    })
    return {"all": None, "completed_only": completed,
            "successful_fulfilment": successful, "broader_intent": intent}


def sample_product2vec_configurations(n_trials=20, random_state=RANDOM_STATE):
    """Include the current baseline then draw unique reproducible configurations."""
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    rng = np.random.default_rng(random_state)
    configurations = [dict(SELECTED_PRODUCT2VEC_CONFIG)]
    seen = {tuple(configurations[0].items())}
    while len(configurations) < n_trials:
        config = {name: values[int(rng.integers(len(values)))]
                  for name, values in P2V_SEARCH_SPACE.items()}
        signature = tuple(config.items())
        if signature not in seen:
            seen.add(signature)
            configurations.append(config)
    return configurations


def blend_weight_grid(include_metadata=False):
    """Return coarse, valid weight combinations that sum exactly to one."""
    values = [value / 10 for value in range(11)]
    rows = []
    metadata_values = [0.0, 0.1, 0.2, 0.3] if include_metadata else [0.0]
    for copurchase in values[:6]:
        for product2vec in values[3:10]:
            for adamic_adar in values[:5]:
                for metadata in metadata_values:
                    if abs(copurchase + product2vec + adamic_adar + metadata - 1.0) < 1e-9:
                        rows.append({"copurchase": copurchase, "product2vec": product2vec,
                                     "adamic_adar": adamic_adar, "metadata": metadata})
    return rows


def _filter_statuses(frame, allowed):
    return frame.copy() if allowed is None else frame[frame["Order Status"].isin(allowed)].copy()


def _fit_bundle(training_orders, p2v_config, graph_config, random_state=RANDOM_STATE):
    catalog = create_product_catalog(training_orders)
    copurchase = CoPurchaseRecommender(
        weighting=graph_config["weighting"],
        min_pair_count=graph_config["min_pair_count"],
        half_life_months=graph_config["half_life_months"],
        ranking=graph_config.get("ranking", "graph"),
    ).fit(training_orders, catalog)
    product2vec = Product2VecRecommender(
        **p2v_config, random_state=random_state
    ).fit(product_catalog=catalog, graph=copurchase.graph)
    adamic_adar = AdamicAdarRecommender().fit(graph=copurchase.graph)
    metadata = MetadataSimilarityRecommender().fit(catalog)
    return {"catalog": catalog, "copurchase": copurchase, "product2vec": product2vec,
            "adamic_adar": adamic_adar, "metadata": metadata}


def _engine(bundle, weights):
    return HybridRecommendationEngine(
        bundle["copurchase"], bundle["product2vec"], bundle["adamic_adar"],
        bundle["catalog"], weights=weights, metadata_model=bundle["metadata"],
    )


def _heterogeneous_bundle(base_bundle, p2v_config, relationship_weights,
                          random_state=RANDOM_STATE):
    config = {**p2v_config, "random_state": random_state}
    heterogeneous = HeterogeneousProduct2VecRecommender(
        relationship_weights=relationship_weights, **config
    ).fit(base_bundle["copurchase"].graph, base_bundle["catalog"])
    return {**base_bundle, "product2vec": heterogeneous}


def _evaluate(bundle, weights, evaluation_orders, training_orders, queries,
              model_name, k=K):
    summary, outcomes = evaluate_cart_engine(
        _engine(bundle, weights), evaluation_orders, training_orders, k=k,
        max_queries=None, random_state=RANDOM_STATE, queries=queries,
    )
    row = summary.iloc[0].to_dict()
    model = bundle["product2vec"]
    if isinstance(model, HeterogeneousProduct2VecRecommender):
        p2v_config = model.product2vec_config
    else:
        p2v_config = {name: getattr(model, name) for name in P2V_SEARCH_SPACE}
    graph = bundle["copurchase"].graph
    row.update({"model": model_name, **{f"weight_{name}": value
                                       for name, value in weights.items()}})
    row.update({
        "random_seed": RANDOM_STATE,
        "training_start": str(pd.to_datetime(training_orders["Order Date"]).min()),
        "training_end": str(pd.to_datetime(training_orders["Order Date"]).max()),
        "evaluation_start": str(pd.to_datetime(evaluation_orders["Order Date"]).min()),
        "evaluation_end": str(pd.to_datetime(evaluation_orders["Order Date"]).max()),
        "training_statuses": " | ".join(sorted(
            training_orders["Order Status"].dropna().astype(str).unique()
        )),
        "graph_weighting": graph.graph.get("weighting", "cosine"),
        "edge_threshold": graph.graph.get("min_pair_count", 1),
        "half_life_months": graph.graph.get("half_life_months"),
        "product2vec_config": json.dumps(p2v_config, sort_keys=True),
        "blend_weights": json.dumps(weights, sort_keys=True),
        "metadata_weights": json.dumps(DEFAULT_METADATA_WEIGHTS, ensure_ascii=False,
                                       sort_keys=True),
        "evaluation_k": k,
    })
    return row, outcomes


def _best(frame):
    return frame.sort_values(
        ["hit_rate_at_k", "mrr_at_k", "recall_at_k", "catalog_coverage_at_k"],
        ascending=False,
    ).iloc[0]


def _graph_record(graph):
    multi = sum(1 for node in graph if graph.degree(node) > 0)
    return {"graph_nodes": graph.number_of_nodes(), "graph_edges": graph.number_of_edges(),
            "connected_nodes": multi}


def _write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _experiment_context(train, dev, graph_config, p2v_config, status_policy,
                        weights, evaluated_queries):
    return {
        "random_seed": RANDOM_STATE,
        "training_start": str(pd.to_datetime(train["Order Date"]).min()),
        "training_end": str(pd.to_datetime(train["Order Date"]).max()),
        "development_start": str(pd.to_datetime(dev["Order Date"]).min()),
        "development_end": str(pd.to_datetime(dev["Order Date"]).max()),
        "status_policy": status_policy,
        "graph_weighting": graph_config["weighting"],
        "edge_threshold": graph_config["min_pair_count"],
        "copurchase_ranking": graph_config.get("ranking", "graph"),
        "half_life_months": graph_config["half_life_months"],
        "product2vec_config": json.dumps(p2v_config, sort_keys=True),
        "blend_weights": json.dumps(weights, sort_keys=True),
        "metadata_weights": json.dumps(DEFAULT_METADATA_WEIGHTS, ensure_ascii=False,
                                       sort_keys=True),
        "evaluation_k": K,
        "evaluated_queries": evaluated_queries,
    }


def run_development_search(data_path=DATA_PATH, output_root=OUTPUT_ROOT,
                           p2v_trials=20, max_queries=250,
                           random_state=RANDOM_STATE):
    """Run all architecture selection on train/development data only."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    orders = load_orders(data_path)
    train, dev, test = chronological_order_split(orders)
    _write_csv(split_summary(train, dev, test), output_root / "split_summary.csv")
    statuses = orders["Order Status"].value_counts().rename_axis("status").reset_index(name="lines")
    _write_csv(statuses, output_root / "observed_statuses.csv")
    dev_queries = leave_one_out_queries(
        dev, set(train["SKU"].astype(str)), max_queries=max_queries, random_state=random_state
    )
    base_graph = {"weighting": "cosine", "min_pair_count": 1,
                  "half_life_months": None, "ranking": "graph"}
    current_p2v = dict(SELECTED_PRODUCT2VEC_CONFIG)
    base_bundle = _fit_bundle(train, current_p2v, base_graph, random_state)

    baseline_specs = {
        "copurchase": {"copurchase": 1.0},
        "product2vec": {"product2vec": 1.0},
        "product2vec_adamic_adar": {"product2vec": 0.75, "adamic_adar": 0.25},
    }
    baseline_rows = []
    for name, weights in baseline_specs.items():
        row, _ = _evaluate(base_bundle, weights, dev, train, dev_queries, name)
        baseline_rows.append(row)
    baseline = pd.DataFrame(baseline_rows)
    _write_csv(baseline, output_root / "baseline" / "baseline_results.csv")

    blend_rows = []
    for weights in blend_weight_grid():
        row, _ = _evaluate(base_bundle, weights, dev, train, dev_queries,
                           "copurchase_product2vec_adamic_adar")
        blend_rows.append(row)
    blend_search = pd.DataFrame(blend_rows)
    _write_csv(blend_search, output_root / "blend_search" / "initial_blend_weight_search.csv")
    best_blend_row = _best(blend_search)
    behavioral_weights = {name: float(best_blend_row[f"weight_{name}"])
                          for name in ("copurchase", "product2vec", "adamic_adar")}

    policies = inspect_status_policies(orders)
    status_rows = []
    for policy_name, allowed in policies.items():
        policy_train = _filter_statuses(train, allowed)
        bundle = _fit_bundle(policy_train, current_p2v, base_graph, random_state)
        row, _ = _evaluate(bundle, behavioral_weights, dev, policy_train, dev_queries,
                           f"status_{policy_name}")
        basket_sizes = policy_train.groupby("Order ID")["SKU"].nunique()
        row.update({"status_policy": policy_name,
                    "allowed_statuses": " | ".join(sorted(allowed)) if allowed else "all",
                    "training_orders": policy_train["Order ID"].nunique(),
                    "multi_item_baskets": int((basket_sizes >= 2).sum()),
                    **_graph_record(bundle["copurchase"].graph)})
        status_rows.append(row)
    status_results = pd.DataFrame(status_rows)
    _write_csv(status_results, output_root / "status_policy" / "status_policy_results.csv")
    selected_policy = str(_best(status_results)["status_policy"])
    selected_allowed = policies[selected_policy]
    selected_train = _filter_statuses(train, selected_allowed)

    graph_candidates = [
        {"weighting": weighting, "min_pair_count": threshold,
         "half_life_months": None, "ranking": "graph"}
        for weighting in ("cosine", "jaccard", "lift") for threshold in (1, 2, 3)
    ] + [{"weighting": "cosine", "min_pair_count": 1,
          "half_life_months": None, "ranking": "confidence"}]
    graph_rows = []
    for config in graph_candidates:
        bundle = _fit_bundle(selected_train, current_p2v, config, random_state)
        row, _ = _evaluate(bundle, behavioral_weights, dev, selected_train, dev_queries,
                           "graph_candidate")
        row.update(config)
        row.update(_graph_record(bundle["copurchase"].graph))
        graph_rows.append(row)
    graph_results = pd.DataFrame(graph_rows)
    _write_csv(graph_results, output_root / "graph_weights" / "graph_weight_results.csv")
    best_graph = _best(graph_results)
    selected_graph = {"weighting": str(best_graph["weighting"]),
                      "min_pair_count": int(best_graph["min_pair_count"]),
                      "half_life_months": None,
                      "ranking": str(best_graph["ranking"])}

    decay_rows = []
    for half_life in HALF_LIVES:
        config = {**selected_graph, "half_life_months": half_life}
        bundle = _fit_bundle(selected_train, current_p2v, config, random_state)
        row, _ = _evaluate(bundle, behavioral_weights, dev, selected_train, dev_queries,
                           "time_decay_candidate")
        row.update({"half_life_months": half_life if half_life is not None else "none",
                    **_graph_record(bundle["copurchase"].graph)})
        decay_rows.append(row)
    decay_results = pd.DataFrame(decay_rows)
    _write_csv(decay_results, output_root / "time_decay" / "time_decay_results.csv")
    best_decay = _best(decay_results)["half_life_months"]
    selected_graph["half_life_months"] = None if str(best_decay) == "none" else float(best_decay)

    p2v_rows = []
    graph_model = CoPurchaseRecommender(
        weighting=selected_graph["weighting"],
        min_pair_count=selected_graph["min_pair_count"],
        half_life_months=selected_graph["half_life_months"],
        ranking=selected_graph["ranking"],
    ).fit(selected_train, create_product_catalog(selected_train))
    for trial, config in enumerate(sample_product2vec_configurations(p2v_trials, random_state), 1):
        print(f"Product2Vec development trial {trial}/{p2v_trials}: {config}", flush=True)
        catalog = graph_model.product_catalog
        product2vec = Product2VecRecommender(**config, random_state=random_state).fit(
            product_catalog=catalog, graph=graph_model.graph
        )
        bundle = {"catalog": catalog, "copurchase": graph_model,
                  "product2vec": product2vec,
                  "adamic_adar": AdamicAdarRecommender().fit(graph=graph_model.graph),
                  "metadata": MetadataSimilarityRecommender().fit(catalog)}
        row, _ = _evaluate(bundle, behavioral_weights, dev, selected_train, dev_queries,
                           "product2vec_candidate")
        row.update({"trial": trial, **config})
        p2v_rows.append(row)
    p2v_results = pd.DataFrame(p2v_rows)
    _write_csv(p2v_results, output_root / "product2vec_search" /
               "product2vec_search_results.csv")
    _write_csv(p2v_results, output_root / "final" / "hyperparameter_search.csv")
    best_p2v_row = _best(p2v_results)
    selected_p2v = {name: (int(best_p2v_row[name]) if name in {
        "n_components", "walk_length", "walks_per_node", "window_size",
        "negative_samples", "epochs"} else float(best_p2v_row[name]))
        for name in P2V_SEARCH_SPACE}

    final_bundle = _fit_bundle(selected_train, selected_p2v, selected_graph, random_state)
    final_blend_rows = []
    for weights in blend_weight_grid(include_metadata=True):
        row, _ = _evaluate(final_bundle, weights, dev, selected_train, dev_queries,
                           "final_weight_candidate")
        final_blend_rows.append(row)
    final_blend = pd.DataFrame(final_blend_rows)
    _write_csv(final_blend, output_root / "blend_search" / "blend_weight_search.csv")
    _write_csv(final_blend, output_root / "final" / "blend_weight_search.csv")
    best_weights_row = _best(final_blend)
    selected_weights = {name: float(best_weights_row[f"weight_{name}"])
                        for name in ("copurchase", "product2vec", "adamic_adar", "metadata")}
    product_graph_weights = dict(selected_weights)

    relationship_candidates = [
        {"copurchase": 1.0, "category": 0.05, "brand": 0.05, "age": 0.05, "hero": 0.05},
        {"copurchase": 1.0, "category": 0.15, "brand": 0.10, "age": 0.15, "hero": 0.10},
        {"copurchase": 1.0, "category": 0.20, "brand": 0.10, "age": 0.20, "hero": 0.10},
        {"copurchase": 1.0, "category": 0.20, "brand": 0.20, "age": 0.20, "hero": 0.20},
    ]
    heterogeneous_rows, heterogeneous_bundles = [], []
    for trial, relationship_weights in enumerate(relationship_candidates, 1):
        print(f"Heterogeneous graph development trial {trial}/{len(relationship_candidates)}",
              flush=True)
        bundle = _heterogeneous_bundle(final_bundle, selected_p2v,
                                       relationship_weights, random_state)
        row, _ = _evaluate(bundle, selected_weights, dev, selected_train, dev_queries,
                           "heterogeneous_graph_candidate")
        row.update({"trial": trial, **{f"relationship_{key}": value
                                       for key, value in relationship_weights.items()}})
        row["heterogeneous_nodes"] = bundle["product2vec"].heterogeneous_graph.number_of_nodes()
        row["heterogeneous_edges"] = bundle["product2vec"].heterogeneous_graph.number_of_edges()
        heterogeneous_rows.append(row)
        heterogeneous_bundles.append(bundle)
    heterogeneous_results = pd.DataFrame(heterogeneous_rows)
    _write_csv(heterogeneous_results, output_root / "heterogeneous_graph" /
               "relationship_weight_search.csv")
    best_heterogeneous_row = _best(heterogeneous_results)
    best_heterogeneous_index = int(best_heterogeneous_row["trial"]) - 1
    selected_relationship_weights = relationship_candidates[best_heterogeneous_index]
    best_heterogeneous_bundle = heterogeneous_bundles[best_heterogeneous_index]
    heterogeneous_blend_rows = []
    for weights in blend_weight_grid(include_metadata=True):
        row, _ = _evaluate(best_heterogeneous_bundle, weights, dev, selected_train,
                           dev_queries, "heterogeneous_weight_candidate")
        heterogeneous_blend_rows.append(row)
    heterogeneous_blend = pd.DataFrame(heterogeneous_blend_rows)
    _write_csv(heterogeneous_blend, output_root / "heterogeneous_graph" /
               "blend_weight_search.csv")
    best_heterogeneous_blend = _best(heterogeneous_blend)
    heterogeneous_weights = {name: float(best_heterogeneous_blend[f"weight_{name}"])
                             for name in ("copurchase", "product2vec",
                                          "adamic_adar", "metadata")}
    use_heterogeneous = tuple(best_heterogeneous_blend[key] for key in
        ("hit_rate_at_k", "mrr_at_k", "recall_at_k", "catalog_coverage_at_k")) > tuple(
        best_weights_row[key] for key in
        ("hit_rate_at_k", "mrr_at_k", "recall_at_k", "catalog_coverage_at_k"))
    if use_heterogeneous:
        selected_weights = heterogeneous_weights
    selected_metric_row = (best_heterogeneous_blend if use_heterogeneous
                           else best_weights_row)

    no_decay_graph = {**selected_graph, "half_life_months": None}
    no_decay_bundle = _fit_bundle(selected_train, selected_p2v, no_decay_graph, random_state)
    ablations = {
        "A Co-purchase": {"copurchase": 1.0},
        "B Product2Vec": {"product2vec": 1.0},
        "C Product2Vec + Adamic-Adar": {"product2vec": 0.75, "adamic_adar": 0.25},
        "D Co-purchase + Product2Vec + Adamic-Adar": behavioral_weights,
        "E D + metadata similarity": product_graph_weights,
        "F E + selected time decay": product_graph_weights,
        "G Heterogeneous Product2Vec": heterogeneous_weights,
    }
    ablation_rows, segment_frames = [], []
    for name, weights in ablations.items():
        if name.startswith(("A ", "B ", "C ", "D ")):
            bundle = base_bundle
        elif name.startswith("E "):
            bundle = no_decay_bundle
        elif name.startswith("G "):
            bundle = best_heterogeneous_bundle
        else:
            bundle = final_bundle
        row, outcomes = _evaluate(bundle, weights, dev, selected_train, dev_queries, name)
        row["split"] = "development"
        ablation_rows.append(row)
        segments = segment_errors(outcomes)
        segments.insert(0, "model", name)
        segments.insert(1, "split", "development")
        segment_frames.append(segments)
    ablation_results = pd.DataFrame(ablation_rows)
    segment_results = pd.concat(segment_frames, ignore_index=True)
    _write_csv(ablation_results, output_root / "final" / "ablation_results.csv")
    _write_csv(segment_results, output_root / "final" / "segment_results.csv")

    best_configuration = {
        "selection_split": "development",
        "random_seed": random_state,
        "k": K,
        "status_policy": selected_policy,
        "allowed_statuses": sorted(selected_allowed) if selected_allowed is not None else None,
        "graph": selected_graph,
        "product2vec": selected_p2v,
        "blend_weights": selected_weights,
        "metadata_field_weights": DEFAULT_METADATA_WEIGHTS,
        "architecture": "heterogeneous" if use_heterogeneous else "product_graph",
        "heterogeneous_relationship_weights": (
            selected_relationship_weights if use_heterogeneous else None
        ),
        "development_queries": len(dev_queries),
        "development_metrics": {
            key: float(selected_metric_row[key]) for key in
            ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr_at_k",
             "catalog_coverage_at_k")
        },
    }
    context = _experiment_context(selected_train, dev, selected_graph, selected_p2v,
                                  selected_policy, selected_weights, len(dev_queries))
    best_configuration["experiment_record"] = context
    config_path = output_root / "final" / "best_dev_configuration.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(best_configuration, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("Development selection complete. Test data has not been evaluated.", flush=True)
    print(json.dumps(best_configuration, ensure_ascii=True, indent=2), flush=True)
    return best_configuration


def run_final_evaluation(data_path=DATA_PATH, output_root=OUTPUT_ROOT,
                         max_queries=250, bootstrap_samples=2000,
                         random_state=RANDOM_STATE):
    """Refit the frozen winner and evaluate the chronological test split once."""
    output_root = Path(output_root)
    final_directory = output_root / "final"
    result_path = final_directory / "final_test_results.csv"
    if result_path.exists():
        raise RuntimeError(
            f"{result_path} already exists; refusing to evaluate the test set again."
        )
    config_path = final_directory / "best_dev_configuration.json"
    if not config_path.exists():
        raise FileNotFoundError("Run the development stage before finalization.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    orders = load_orders(data_path)
    train, dev, test = chronological_order_split(orders)
    train_dev_all = pd.concat([train, dev], ignore_index=True)
    allowed = config["allowed_statuses"]
    training = _filter_statuses(train_dev_all, set(allowed) if allowed is not None else None)
    graph_config = dict(config["graph"])
    bundle = _fit_bundle(training, config["product2vec"], graph_config, random_state)
    if config.get("architecture") == "heterogeneous":
        bundle = _heterogeneous_bundle(
            bundle, config["product2vec"], config["heterogeneous_relationship_weights"],
            random_state,
        )
    test_queries = leave_one_out_queries(
        test, set(train_dev_all["SKU"].astype(str)), max_queries=max_queries,
        random_state=random_state,
    )
    final_row, outcomes = _evaluate(
        bundle, config["blend_weights"], test, training, test_queries,
        "selected_final_model",
    )
    final_row.update({"split": "test", "status_policy": config["status_policy"],
                      "graph_weighting": graph_config["weighting"],
                      "edge_threshold": graph_config["min_pair_count"],
                      "half_life_months": graph_config["half_life_months"]})
    _write_csv(pd.DataFrame([final_row]), result_path)
    _write_csv(outcomes, final_directory / "final_test_outcomes.csv")
    segments = segment_errors(outcomes)
    segments.insert(0, "model", "selected_final_model")
    segments.insert(1, "split", "test")
    _write_csv(segments, final_directory / "final_test_segments.csv")
    intervals = bootstrap_ranking_intervals(
        outcomes, n_bootstrap=bootstrap_samples, random_state=random_state
    )
    _write_csv(intervals, final_directory / "bootstrap_intervals.csv")
    historical_path = output_root.parent / "phase6" / "summary_metrics.csv"
    if historical_path.exists():
        historical = pd.read_csv(historical_path).iloc[0]
        metric_columns = ["precision_at_k", "recall_at_k", "hit_rate_at_k",
                          "mrr_at_k", "catalog_coverage_at_k"]
        comparison = pd.DataFrame([
            {"model": "original_product2vec_adamic_adar",
             **{column: historical[column] for column in metric_columns}},
            {"model": "selected_final_model",
             **{column: final_row[column] for column in metric_columns}},
        ])
        for column in metric_columns:
            comparison[f"delta_vs_original_{column}"] = (
                comparison[column] - float(historical[column])
            )
        _write_csv(comparison, final_directory / "baseline_comparison.csv")
    print("Frozen final model evaluated on test exactly once.", flush=True)
    print(pd.DataFrame([final_row])[[
        "model", "evaluated_orders", "hit_rate_at_k", "recall_at_k",
        "mrr_at_k", "catalog_coverage_at_k",
    ]].to_string(index=False), flush=True)
    return pd.DataFrame([final_row]), outcomes, intervals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("development", "finalize"))
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=250)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main():
    arguments = parse_args()
    if arguments.stage == "development":
        run_development_search(arguments.data, arguments.output, arguments.trials,
                               arguments.max_queries, arguments.seed)
    else:
        run_final_evaluation(arguments.data, arguments.output, arguments.max_queries,
                             arguments.bootstrap_samples, arguments.seed)


if __name__ == "__main__":
    main()
