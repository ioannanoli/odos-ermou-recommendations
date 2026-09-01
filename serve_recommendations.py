"""Train and query the frozen Odos Ermou recommendation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from src.data_loader import load_orders
from src.final_recommender import (DEFAULT_CONFIG_PATH, FinalRecommender)
from src.model_export import export_model_parameters


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
MODEL_PATH = Path("models/final_recommender.pkl")
MODEL_WEIGHTS_PATH = Path("model_weights")


def load_available_skus(path, sku_column="SKU"):
    """Read the set of sellable SKUs from a CSV or Excel inventory file."""
    path = Path(path)
    if path.suffix.casefold() in {".xlsx", ".xls"}:
        inventory = pd.read_excel(path, dtype={sku_column: str})
    else:
        inventory = pd.read_csv(path, dtype={sku_column: "string"})
    if sku_column not in inventory:
        raise ValueError(f"Inventory column not found: {sku_column}")
    return set(inventory[sku_column].dropna().astype(str).str.strip())


def train_model(data_path=DATA_PATH, config_path=DEFAULT_CONFIG_PATH,
                model_path=MODEL_PATH):
    """Fit the frozen architecture on all historical data and persist it."""
    model = FinalRecommender.from_config_path(config_path)
    model.fit(load_orders(data_path))
    model.save(model_path)
    return model


def recommend_from_model(model_path, cart_skus, top_n=10, inventory_path=None,
                         inventory_column="SKU", output_path=None):
    """Load the trained model and return inventory-aware cart recommendations."""
    model = FinalRecommender.load(model_path)
    available = (load_available_skus(inventory_path, inventory_column)
                 if inventory_path else None)
    recommendations = model.recommend(cart_skus, top_n=top_n, available_skus=available)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recommendations.to_csv(output_path, index=False, encoding="utf-8-sig")
    return recommendations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train and save the frozen final model.")
    train.add_argument("--data", type=Path, default=DATA_PATH)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    train.add_argument("--model", type=Path, default=MODEL_PATH)

    recommend = subparsers.add_parser("recommend", help="Recommend for one cart.")
    recommend.add_argument("skus", nargs="+", help="One or more cart SKUs.")
    recommend.add_argument("--model", type=Path, default=MODEL_PATH)
    recommend.add_argument("--top-n", type=int, default=10)
    recommend.add_argument("--inventory", type=Path)
    recommend.add_argument("--inventory-column", default="SKU")
    recommend.add_argument("--output", type=Path)

    inspect = subparsers.add_parser("inspect", help="Show fitted model details.")
    inspect.add_argument("--model", type=Path, default=MODEL_PATH)

    export = subparsers.add_parser(
        "export-weights",
        help="Export graph edges and final Node2vec vectors as readable files.",
    )
    export.add_argument("--model", type=Path, default=MODEL_PATH)
    export.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    export.add_argument("--output", type=Path, default=MODEL_WEIGHTS_PATH)
    return parser


def main():
    """Console entry point for training and serving the final recommender."""
    sys.stdout.reconfigure(errors="backslashreplace")
    arguments = build_parser().parse_args()
    if arguments.command == "train":
        model = train_model(arguments.data, arguments.config, arguments.model)
        print(f"Saved final model to {arguments.model}")
        print(json.dumps(model.training_summary, ensure_ascii=True, indent=2))
    elif arguments.command == "recommend":
        recommendations = recommend_from_model(
            arguments.model, arguments.skus, arguments.top_n,
            arguments.inventory, arguments.inventory_column, arguments.output,
        )
        print(recommendations.to_string(index=False))
        if arguments.output:
            print(f"\nSaved recommendations to {arguments.output}")
    elif arguments.command == "inspect":
        model = FinalRecommender.load(arguments.model)
        print(json.dumps(model.training_summary, ensure_ascii=True, indent=2))
        print(json.dumps(model.configuration["blend_weights"], indent=2))
    else:
        model = FinalRecommender.load(arguments.model)
        manifest = export_model_parameters(
            model, arguments.output, arguments.model, arguments.config
        )
        print(f"Exported learned parameters to {arguments.output.resolve()}")
        print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
