from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.data_loader import PRODUCT_COLUMNS
from src.final_recommender import FinalRecommender, load_frozen_configuration
from src.metadata_recommender import DEFAULT_METADATA_WEIGHTS
from serve_recommendations import load_available_skus
from experiments.candidate_health_report import audit_candidate_health
from src.logistic_ranker import CandidateFeatureBuilder, FEATURE_NAMES
from src.model_export import export_model_parameters


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
            ROOT / "model_configs/historical_basket_model.json"
        )
        self.assertEqual(configuration["architecture"], "heterogeneous")
        self.assertEqual(configuration["blend_weights"]["copurchase"], 0.4)

    def test_product_page_configuration_uses_tfidf(self):
        configuration = load_frozen_configuration()
        self.assertEqual(configuration["blend_weights"]["copurchase"], 0.4)
        self.assertEqual(configuration["blend_weights"]["text"], 0.4)
        self.assertEqual(configuration["selection_split"], "development_single_sku")

    def test_fit_recommend_inventory_and_enrichment(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        recommendations = model.recommend(["A"], top_n=2)
        self.assertEqual(recommendations.iloc[0]["recommended_sku"], "B")
        self.assertIn("Product Name", recommendations.columns)
        self.assertIn("text_score", recommendations.columns)
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

    def test_product_page_recommendation_views(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        bought_together = model.recommend_frequently_bought_together("A", top_n=2)
        similar = model.recommend_similar("A", top_n=2)
        self.assertEqual(bought_together.iloc[0]["recommended_sku"], "B")
        self.assertNotIn("A", bought_together["recommended_sku"].tolist())
        self.assertNotIn("A", similar["recommended_sku"].tolist())
        self.assertTrue((bought_together["product2vec_score"] == 0).all())
        self.assertTrue((similar["copurchase_score"] == 0).all())

    def test_product_page_views_apply_inventory(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        result = model.recommend_frequently_bought_together(
            "A", top_n=2, available_skus={"C"}
        )
        self.assertEqual(result["recommended_sku"].tolist(), ["C"])

    def test_inventory_loader_preserves_sku_text(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.csv"
            pd.DataFrame({"SKU": ["0012", "A-3"]}).to_csv(path, index=False)
            self.assertEqual(load_available_skus(path), {"0012", "A-3"})

    def test_order_loader_accepts_csv_and_preserves_leading_zero_sku(self):
        from src.data_loader import load_orders

        with TemporaryDirectory() as directory:
            path = Path(directory) / "orders.csv"
            pd.DataFrame({
                "Order ID": [1],
                "Order Date": ["2026-08-01 10:00:00"],
                "Order Status": ["wc-cancelled"],
                "SKU": ["0012"],
                "Product Name": ["Test"],
            }).to_csv(path, index=False, encoding="utf-8-sig")
            loaded = load_orders(path)
            self.assertEqual(loaded.iloc[0]["SKU"], "0012")
            self.assertEqual(loaded.iloc[0]["Order Status"], "wc-cancelled")

    def test_unlabeled_candidate_health_report_checks_every_catalog_sku(self):
        configuration = {
            **self.configuration,
            "candidate_generation": {
                "minimum_per_source": 3,
                "multiplier": 2,
                "maximum_per_source": 10,
                "track_sources": True,
            },
        }
        model = FinalRecommender(configuration).fit(self.orders)
        summary, details, segments, sources, combinations = audit_candidate_health(
            model, top_n=2, check_determinism=True, progress_every=0
        )
        self.assertEqual(summary["evaluated_skus"], 3)
        self.assertEqual(summary["failures"], 0)
        self.assertEqual(summary["invariant_failures"], 0)
        self.assertFalse(summary["accuracy_metrics_calculated"])
        self.assertEqual(set(details["sku"]), {"A", "B", "C"})
        self.assertTrue((details["seed_excluded"] == 1).all())
        self.assertIn("rare", set(segments["segment"]))
        self.assertIn("copurchase", set(sources["source"]))
        self.assertFalse(combinations.empty)
        pool = model.engine.candidate_pool(["A"], top_n=2)
        features = CandidateFeatureBuilder(model, self.orders).transform("A", pool)
        self.assertEqual(features.shape, (len(pool), len(FEATURE_NAMES)))

    def test_learned_parameter_export_contains_graph_and_embeddings(self):
        model = FinalRecommender(self.configuration).fit(self.orders)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = export_model_parameters(model, output)
            nodes = pd.read_csv(output / "final_graph_nodes.csv")
            edges = pd.read_csv(output / "final_graph_edges.csv")
            embeddings = pd.read_csv(output / "final_node2vec_embeddings.csv")
            self.assertEqual(manifest["graph"]["nodes"], len(nodes))
            self.assertEqual(manifest["graph"]["edges"], len(edges))
            self.assertEqual(manifest["node2vec"]["nodes"], len(embeddings))
            self.assertIn("selected_edge_weight", edges)
            self.assertIn("left_to_right_confidence", edges)
            self.assertIn("embedding_00", embeddings)


if __name__ == "__main__":
    unittest.main()
