import unittest

import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender, build_copurchase_graph
from src.evaluation import chronological_order_split, evaluate_recommender, ranking_metrics
from src.fp_growth import frequent_itemsets
from src.product2vec_recommender import Product2VecRecommender
from phase4_experiments import sample_configurations


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.orders = pd.DataFrame({
            "Order ID": [1, 1, 2, 2, 3, 3, 3],
            "SKU": ["A", "B", "A", "B", "A", "B", "C"],
        })

    def test_normalized_graph(self):
        graph = build_copurchase_graph(self.orders)
        self.assertEqual(graph["A"]["B"]["count"], 3)
        self.assertAlmostEqual(graph["A"]["B"]["weight"], 1.0)
        self.assertEqual(CoPurchaseRecommender().fit(self.orders).recommend("A", 1)
                         .iloc[0]["recommended_sku"], "B")

    def test_fp_growth(self):
        itemsets = frequent_itemsets(self.orders, min_support=2)
        counts = dict(zip(itemsets["itemset"], itemsets["count"]))
        self.assertEqual(counts[("A", "B")], 3)
        self.assertNotIn(("C",), counts)

    def test_node2vec_is_reproducible(self):
        options = dict(n_components=8, walk_length=5, walks_per_node=2,
                       negative_samples=2, epochs=2, random_state=7)
        first = Product2VecRecommender(**options).fit(self.orders)
        second = Product2VecRecommender(**options).fit(self.orders)
        self.assertEqual(first.recommend("A", 2).to_dict("records"),
                         second.recommend("A", 2).to_dict("records"))

    def test_chronological_split_keeps_orders_intact(self):
        dated = pd.DataFrame({
            "Order ID": [3, 1, 2, 2, 4],
            "Order Date": ["2024-01-03", "2024-01-01", "2024-01-02",
                           "2024-01-02", "2024-01-04"],
            "SKU": ["C", "A", "A", "B", "D"],
        })
        train, dev, test = chronological_order_split(
            dated, train_fraction=0.5, dev_fraction=0.25
        )
        split_ids = [set(frame["Order ID"]) for frame in (train, dev, test)]
        self.assertEqual(split_ids, [{1, 2}, {3}, {4}])
        self.assertLessEqual(train["Order Date"].max(), dev["Order Date"].min())
        self.assertLessEqual(dev["Order Date"].max(), test["Order Date"].min())

    def test_ranking_metrics(self):
        metrics = ranking_metrics(["X", "B", "C"], {"B", "C"}, k=3)
        self.assertAlmostEqual(metrics["precision_at_k"], 2 / 3)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 0.5)

    def test_evaluation_counts_eligible_queries(self):
        model = CoPurchaseRecommender().fit(self.orders)
        metrics = evaluate_recommender(model, self.orders, k=2)
        self.assertEqual(metrics["evaluated_queries"], 7)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)

    def test_random_search_is_reproducible_and_unique(self):
        first = sample_configurations(4, random_state=9)
        second = sample_configurations(4, random_state=9)
        self.assertEqual(first, second)
        self.assertEqual(len({tuple(config.items()) for config in first}), 4)


if __name__ == "__main__":
    unittest.main()
