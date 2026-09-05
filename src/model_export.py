"""Readable exports of learned graph and Product2Vec model parameters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_freeze import DEFAULT_CONFIG_PATH, sha256_file


def graph_node_frame(model):
    """Return one row per fitted product-graph node."""
    graph = model.engine.copurchase_model.graph
    rows = []
    for raw_sku, attributes in sorted(graph.nodes(data=True), key=lambda item: str(item[0])):
        rows.append({
            "sku": str(raw_sku),
            "order_count": int(attributes.get("order_count", 0)),
            "weighted_order_count": float(
                attributes.get("weighted_order_count", attributes.get("order_count", 0))
            ),
            "degree": int(graph.degree(raw_sku)),
            "weighted_degree": float(graph.degree(raw_sku, weight="weight")),
        })
    return pd.DataFrame(rows, columns=[
        "sku", "order_count", "weighted_order_count", "degree", "weighted_degree",
    ])


def graph_edge_frame(model):
    """Return all fitted co-purchase edges and their learned statistics."""
    graph = model.engine.copurchase_model.graph
    effective_orders = max(float(graph.graph.get("effective_orders", 0)), 1.0)
    rows = []
    for raw_left, raw_right, attributes in graph.edges(data=True):
        left, right = sorted((str(raw_left), str(raw_right)))
        left_data, right_data = graph.nodes[left], graph.nodes[right]
        pair_weight = float(
            attributes.get("weighted_count", attributes.get("count", 0))
        )
        left_orders = float(
            left_data.get("weighted_order_count", left_data.get("order_count", 0))
        )
        right_orders = float(
            right_data.get("weighted_order_count", right_data.get("order_count", 0))
        )
        rows.append({
            "left_sku": left,
            "right_sku": right,
            "raw_pair_count": int(attributes.get("count", 0)),
            "weighted_pair_count": pair_weight,
            "support": pair_weight / effective_orders,
            "cosine": float(attributes.get("cosine", 0.0)),
            "jaccard": float(attributes.get("jaccard", 0.0)),
            "lift": float(attributes.get("lift", 0.0)),
            "selected_edge_weight": float(attributes.get("weight", 0.0)),
            "left_to_right_confidence": pair_weight / max(left_orders, 1e-12),
            "right_to_left_confidence": pair_weight / max(right_orders, 1e-12),
        })
    columns = [
        "left_sku", "right_sku", "raw_pair_count", "weighted_pair_count",
        "support", "cosine", "jaccard", "lift", "selected_edge_weight",
        "left_to_right_confidence", "right_to_left_confidence",
    ]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values(["left_sku", "right_sku"]).reset_index(drop=True))


def node2vec_embedding_frame(model):
    """Return the final normalized embedding vector for every walk-graph node."""
    wrapper = model.engine.product2vec_model
    embedding_model = getattr(wrapper, "model", wrapper)
    if embedding_model.embeddings is None:
        raise RuntimeError("The Product2Vec model has no fitted embeddings.")
    embeddings = np.asarray(embedding_model.embeddings, dtype=float)
    if embeddings.ndim != 2 or len(embedding_model.index_to_sku) != len(embeddings):
        raise ValueError("The Product2Vec index and embedding matrix do not align.")

    metadata_rows = []
    for index in range(len(embeddings)):
        node_id = str(embedding_model.index_to_sku[index])
        if node_id.startswith("product::"):
            node_type, node_value = "product", node_id.split("::", 1)[1]
        elif "::" in node_id:
            node_type, node_value = node_id.split("::", 1)
        else:
            node_type, node_value = "product", node_id
        metadata_rows.append({
            "node_index": index,
            "node_id": node_id,
            "node_type": node_type,
            "node_value": node_value,
        })
    vector_columns = [f"embedding_{index:02d}" for index in range(embeddings.shape[1])]
    return pd.concat([
        pd.DataFrame(metadata_rows),
        pd.DataFrame(embeddings, columns=vector_columns),
    ], axis=1)


def export_model_parameters(model, output_directory, model_path=None,
                            config_path=DEFAULT_CONFIG_PATH):
    """Export scalar-independent learned parameters and a verification manifest."""
    if model.engine is None:
        raise RuntimeError("Fit or load the final recommender before exporting it.")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    frames = {
        "final_graph_nodes.csv": graph_node_frame(model),
        "final_graph_edges.csv": graph_edge_frame(model),
        "final_node2vec_embeddings.csv": node2vec_embedding_frame(model),
    }
    for file_name, frame in frames.items():
        frame.to_csv(output_directory / file_name, index=False, encoding="utf-8-sig")

    embedding_frame = frames["final_node2vec_embeddings.csv"]
    manifest = {
        "schema_version": 1,
        "model_artifact": Path(model_path).name if model_path else None,
        "model_artifact_sha256": (
            sha256_file(model_path) if model_path is not None else None
        ),
        "configuration": Path(config_path).as_posix(),
        "configuration_sha256": sha256_file(config_path),
        "training_summary": model.training_summary,
        "graph": {
            "nodes": int(len(frames["final_graph_nodes.csv"])),
            "edges": int(len(frames["final_graph_edges.csv"])),
            "weighting": model.engine.copurchase_model.weighting,
            "ranking": model.engine.copurchase_model.ranking,
            "edge_weight_column": "selected_edge_weight",
        },
        "node2vec": {
            "nodes": int(len(embedding_frame)),
            "dimensions": int(sum(
                column.startswith("embedding_") for column in embedding_frame.columns
            )),
            "representation": "L2-normalized sum of Skip-Gram input and output vectors",
        },
        "files": {
            file_name: {
                "rows": int(len(frame)),
                "sha256": sha256_file(output_directory / file_name),
            }
            for file_name, frame in frames.items()
        },
    }
    manifest_path = output_directory / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
