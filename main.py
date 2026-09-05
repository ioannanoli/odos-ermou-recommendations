"""Run the data-processing, basket-analysis, and embedding pipeline."""

from pathlib import Path
import sys

from src.copurchase_recommender import CoPurchaseRecommender
from src.data_loader import create_product_catalog, load_orders
from src.fp_growth import frequent_itemsets
from src.model_config import SELECTED_PRODUCT2VEC_CONFIG
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import AdamicAdarRecommender, RecommendationEngine


DATA_PATH = Path("data/Orders-Export-2026-June-07-2054.xlsx")
OUTPUT_DIRECTORY = Path("outputs")
TOP_N = 10
MIN_ITEMSET_SUPPORT = 0.002
def _save_itemsets(orders, output_directory):
    """Mine frequent baskets and save their SKUs in a readable format."""
    itemsets = frequent_itemsets(
        orders,
        min_support=MIN_ITEMSET_SUPPORT,
        max_length=3,
    )
    itemsets["itemset"] = itemsets["itemset"].map(lambda values: " | ".join(values))
    itemsets.to_csv(
        output_directory / "frequent_itemsets.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _save_recommendations(model, model_name, example_sku, product_catalog, output_directory):
    """Generate, enrich, print, and save one model's recommendations."""
    recommendations = model.recommend(example_sku, top_n=TOP_N)
    detailed = recommendations.join(product_catalog, on="recommended_sku")
    detailed.to_csv(
        output_directory / f"{model_name}_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"{model_name.title()} recommendations for {example_sku}:")
    print(recommendations.to_string(index=False))
    print()


def run_pipeline(data_path=DATA_PATH, output_directory=OUTPUT_DIRECTORY):
    """Load the workbook, train Phase 1–5 models, and write CSV results."""
    orders = load_orders(data_path)
    product_catalog = create_product_catalog(orders)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    copurchase_model = CoPurchaseRecommender().fit(orders, product_catalog)
    product2vec_model = Product2VecRecommender(
        **SELECTED_PRODUCT2VEC_CONFIG
    ).fit(orders, product_catalog)
    adamic_adar_model = AdamicAdarRecommender().fit(graph=copurchase_model.graph)
    example_sku = orders["SKU"].value_counts().index[0]

    _save_itemsets(orders, output_directory)
    _save_recommendations(
        copurchase_model,
        "copurchase",
        example_sku,
        product_catalog,
        output_directory,
    )
    _save_recommendations(
        product2vec_model,
        "product2vec",
        example_sku,
        product_catalog,
        output_directory,
    )
    _save_recommendations(
        adamic_adar_model,
        "adamic_adar",
        example_sku,
        product_catalog,
        output_directory,
    )

    direct = copurchase_model.recommend(example_sku, top_n=1)
    example_cart = [example_sku]
    if not direct.empty:
        example_cart.append(direct.iloc[0]["recommended_sku"])
    engine = RecommendationEngine(product2vec_model, adamic_adar_model, product_catalog)
    cart_recommendations = engine.recommend(example_cart, top_n=TOP_N)
    detailed_cart = cart_recommendations.join(product_catalog, on="recommended_sku")
    detailed_cart.to_csv(
        output_directory / "cart_recommendations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Cart recommendations for {' | '.join(example_cart)}:")
    print(cart_recommendations.to_string(index=False))
    print()


def main():
    """Console entry point for the production recommendation pipeline."""
    # Prevent legacy Windows consoles from failing on Greek product text.
    sys.stdout.reconfigure(errors="backslashreplace")
    run_pipeline()


if __name__ == "__main__":
    main()
