from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data_loader import PRODUCT_COLUMNS
from src.final_recommender import FinalRecommender, load_frozen_configuration
from src.metadata_recommender import DEFAULT_METADATA_WEIGHTS
from serve_recommendations import load_available_skus


ROOT = Path(__file__).resolve().parents[1]


class FinalRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.orders = pd.DataFrame({
            "Order ID": [1, 1, 2, 2, 3, 3],
            "Order Date": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-02-01", "2024-02-01",
                "2024-03-01", "2024-03-01",
            ]),
            "Order Status": ["wc-completed"] * 6,
            "SKU": ["A", "B", "A", "C", "A", "B"],
            "Product Name": ["Set A", "Box B", "Set A", "Toy C", "Set A", "Box B"],
            "Κατηγορίες προϊόντων": ["Building", "Storage", "Building", "Dolls",
                                        "Building", "Storage"],
            "Μάρκες": ["Brand A", "Brand B", "Brand A", "Brand C",
                        "Brand A", "Brand B"],
            "Προϊόν Ηλικία": ["6+", "6+", "6+", "3+", "6+", "6+"],
            "Προϊόν Ήρωας": ["City", "Unknown", "City", "Unknown", "City", "Unknown"],
            "Προϊόν Φύλλο": ["Unisex"] * 6,
            "sex_p": ["Unisex"] * 6,
            "hero_p": ["City", "Unknown", "City", "Unknown", "City", "Unknown"],
            "age_p": ["6+", "6+", "6+", "3+", "6+", "6+"],
        })
        self.assertTrue(set(PRODUCT_COLUMNS).issubset(self.orders.columns))
        self.configuration = {
            "random_seed": 3,
            "allowed_statuses": None,
            "graph": {"weighting": "cosine", "min_pair_count": 1,
                      "half_life_months": None, "ranking": "confidence"},
            "product2vec": {"n_components": 8, "walk_length": 4,
                            "walks_per_node": 1, "window_size": 2,
                            "negative_samples": 1, "epochs": 1,
                            "learning_rate": 0.025, "p": 1.0, "q": 1.0},
            "blend_weights": {"copurchase": 1.0, "product2vec": 0.0,
                              "adamic_adar": 0.0, "metadata": 0.0},
            "metadata_field_weights": DEFAULT_METADATA_WEIGHTS,
            "architecture": "product_graph",
            "heterogeneous_relationship_weights": None,
        }

    def test_frozen_configuration_loads(self):
        configuration = load_frozen_configuration(
            ROOT / "outputs/improvement_experiments/final/best_dev_configuration.json"
        )
        self.assertEqual(configuration["architecture"], "heterogeneous")
        self.assertEqual(configuration["blend_weights"]["copurchase"], 0.4)

    def test_fit_recommend_inventory_and_enrichment(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        recommendations = model.recommend(["A"], top_n=2)
        self.assertEqual(recommendations.iloc[0]["recommended_sku"], "B")
        self.assertIn("Product Name", recommendations.columns)
        available = model.recommend(["A"], top_n=2, available_skus={"C"})
        self.assertEqual(available["recommended_sku"].tolist(), ["C"])
        self.assertEqual(model.training_summary["order_count"], 3)

    def test_saved_model_round_trip(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        expected = model.recommend(["A"], top_n=2).to_dict("records")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pkl"
            model.save(path)
            loaded = FinalRecommender.load(path)
            self.assertEqual(loaded.recommend(["A"], top_n=2).to_dict("records"), expected)

    def test_inventory_loader_preserves_sku_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            pd.DataFrame({"SKU": ["0012", "A-3"]}).to_csv(path, index=False)
            self.assertEqual(load_available_skus(path), {"0012", "A-3"})


if __name__ == "__main__":
    unittest.main()
