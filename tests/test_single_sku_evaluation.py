import unittest

import pandas as pd

from single_sku_evaluation import PopularityBaseline, four_signal_weight_grid
from src.evaluation import evaluate_single_sku_engine, single_sku_queries


class _FixedEngine:
    def recommend(self, cart_skus, top_n=10):
        seed = str(cart_skus[0])
        rankings = {"A": ["B", "C"], "B": ["A", "C"]}
        rows = [{"recommended_sku": sku} for sku in rankings.get(seed, [])[:top_n]]
        return pd.DataFrame(rows, columns=["recommended_sku"])


class SingleSkuEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.orders = pd.DataFrame({
            "Order ID": [1, 1, 1, 2, 2, 3],
            "SKU": ["A", "B", "C", "A", "B", "D"],
        })

    def test_queries_use_exactly_one_pair_per_eligible_order(self):
        first = single_sku_queries(
            self.orders, {"A", "B", "C"}, random_state=7
        )
        second = single_sku_queries(
            self.orders, {"A", "B", "C"}, random_state=7
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual({query["order_id"] for query in first}, {"1", "2"})
        for query in first:
            self.assertNotEqual(query["seed_sku"], query["target_sku"])
            self.assertNotIn("D", (query["seed_sku"], query["target_sku"]))

    def test_single_sku_metrics_use_one_hidden_target(self):
        queries = [{
            "order_id": "1", "seed_sku": "A", "target_sku": "B",
            "held_out_basket_size": 3,
        }]
        summary, outcomes = evaluate_single_sku_engine(
            _FixedEngine(), queries, self.orders, {"A", "B", "C"},
            k=2, candidate_k=2,
        )
        self.assertEqual(summary.iloc[0]["hit_rate_at_k"], 1.0)
        self.assertEqual(summary.iloc[0]["precision_at_k"], 0.5)
        self.assertEqual(summary.iloc[0]["mrr_at_k"], 1.0)
        self.assertEqual(outcomes.iloc[0]["target_rank"], 1)

    def test_category_popularity_prioritizes_shared_category(self):
        catalog = pd.DataFrame(
            {"category": ["building", "building", "dolls"]},
            index=["A", "B", "C"],
        )
        baseline = PopularityBaseline(self.orders, catalog, "category")
        result = baseline.recommend(["A"], top_n=2)
        self.assertEqual(result["recommended_sku"].tolist(), ["B", "C"])

    def test_four_signal_grid_keeps_every_signal_active(self):
        weights = four_signal_weight_grid(0.1)
        self.assertEqual(len(weights), 84)
        for row in weights:
            self.assertAlmostEqual(sum(row.values()), 1.0)
            self.assertTrue(all(value >= 0.1 for value in row.values()))


if __name__ == "__main__":
    unittest.main()
