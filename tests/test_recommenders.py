import unittest

import networkx as nx
import numpy as np
import pandas as pd

from src.copurchase_recommender import CoPurchaseRecommender, build_copurchase_graph
from src.evaluation import chronological_order_split, evaluate_recommender, ranking_metrics
from src.error_analysis import (bootstrap_ranking_intervals, leave_one_out_queries,
                                qualitative_samples, softmax_cross_entropy)
from src.fp_growth import frequent_itemsets
from src.heterogeneous_graph import build_heterogeneous_graph
from src.metadata_recommender import (DEFAULT_METADATA_WEIGHTS,
                                      MetadataSimilarityRecommender)
from src.product2vec_recommender import Product2VecRecommender
from src.recommendation_engine import (AdamicAdarRecommender,
                                       HybridRecommendationEngine,
                                       RecommendationEngine, filter_by_metadata)
from src.metadata_transfer_recommender import MetadataEnhancedEngine, MetadataTransferRecommender
from src.time_weighting import order_time_weights
from src.tfidf_recommender import TfidfNameRecommender
from phase4_experiments import sample_configurations
from graph_visualizations import graph_statistics, neighborhood_subgraph, strongest_edge_backbone


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

    def test_graph_threshold_weightings_and_time_decay(self):
        dated = self.orders.copy()
        dated["Order Date"] = pd.to_datetime([
            "2020-01-01", "2020-01-01", "2024-01-01", "2024-01-01",
            "2024-06-01", "2024-06-01", "2024-06-01",
        ])
        thresholded = build_copurchase_graph(dated, weighting="jaccard", min_pair_count=2)
        self.assertTrue(thresholded.has_edge("A", "B"))
        self.assertFalse(thresholded.has_edge("A", "C"))
        self.assertAlmostEqual(thresholded["A"]["B"]["weight"],
                               thresholded["A"]["B"]["jaccard"])
        decayed = build_copurchase_graph(dated, half_life_months=12)
        self.assertLess(decayed["A"]["B"]["weighted_count"], 3)
        weights = order_time_weights(dated, 12)
        self.assertLess(weights.loc[1], weights.loc[3])

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

    def test_node2vec_does_not_depend_on_graph_insertion_order(self):
        edges = [("A", "B"), ("A", "C"), ("B", "C")]
        graphs = []
        for ordered_edges in (edges, list(reversed(edges))):
            graph = nx.Graph()
            for left, right in ordered_edges:
                graph.add_edge(left, right, weight=1.0)
            for node in graph:
                graph.nodes[node]["order_count"] = 2
            graphs.append(graph)
        options = dict(n_components=8, walk_length=5, walks_per_node=2,
                       negative_samples=2, epochs=2, random_state=7)
        first = Product2VecRecommender(**options).fit(graph=graphs[0])
        second = Product2VecRecommender(**options).fit(graph=graphs[1])
        np.testing.assert_allclose(first.embeddings, second.embeddings)

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

    def test_adamic_adar_predicts_missing_link(self):
        import math
        orders = pd.DataFrame({
            "Order ID": [1, 1, 2, 2],
            "SKU": ["A", "X", "B", "X"],
        })
        result = AdamicAdarRecommender().fit(orders=orders).recommend("A")
        self.assertEqual(result.iloc[0]["recommended_sku"], "B")
        self.assertAlmostEqual(result.iloc[0]["adamic_adar_score"], 1 / math.log(2))

    def test_product2vec_recommends_for_cart(self):
        model = Product2VecRecommender(
            n_components=8, walk_length=5, walks_per_node=2,
            negative_samples=2, epochs=2, random_state=7,
        ).fit(self.orders)
        result = model.recommend_cart(["A", "B"], top_n=1)
        self.assertEqual(result.iloc[0]["recommended_sku"], "C")

    def test_metadata_filter_requires_all_fields(self):
        recommendations = pd.DataFrame({
            "recommended_sku": ["A", "B"], "score": [1.0, 0.5]
        })
        catalog = pd.DataFrame({
            "age": ["6+", "3+"], "category": ["LEGO | Building", "Dolls"]
        }, index=["A", "B"])
        result = filter_by_metadata(
            recommendations, catalog, {"age": "6+", "category": "building"}
        )
        self.assertEqual(result["recommended_sku"].tolist(), ["A"])

    def test_phase5_engine_blends_and_filters(self):
        catalog = pd.DataFrame({"category": ["keep", "keep", "keep"]},
                               index=["A", "B", "C"])
        vector_model = Product2VecRecommender(
            n_components=8, walk_length=5, walks_per_node=2,
            negative_samples=2, epochs=2, random_state=7,
        ).fit(self.orders, catalog)
        link_model = AdamicAdarRecommender().fit(graph=vector_model.graph)
        result = RecommendationEngine(vector_model, link_model, catalog).recommend(
            ["A"], top_n=2, metadata_filters={"category": "keep"}
        )
        self.assertFalse(result.empty)
        self.assertNotIn("A", result["recommended_sku"].tolist())
        self.assertIn("recommendation_score", result.columns)

    def test_softmax_cross_entropy(self):
        import math
        self.assertAlmostEqual(softmax_cross_entropy([0.0, 0.0, 0.0], 1), math.log(3))
        self.assertLess(softmax_cross_entropy([0.0, 4.0, 0.0], 1), 0.1)

    def test_leave_one_out_queries_are_reproducible(self):
        first = leave_one_out_queries(self.orders, {"A", "B", "C"}, random_state=5)
        second = leave_one_out_queries(self.orders, {"A", "B", "C"}, random_state=5)
        self.assertEqual(first, second)
        for query in first:
            self.assertNotIn(query["target_sku"], query["cart_skus"])

    def test_qualitative_samples_flag_metadata_mismatch(self):
        outcomes = pd.DataFrame([{
            "target_sku": "A", "top_prediction": "B", "hit_at_k": 0
        }])
        catalog = pd.DataFrame({
            "Product Name": ["Building set", "Baby rattle"],
            "Κατηγορίες προϊόντων": ["Building", "Baby"],
            "Προϊόν Ηλικία": ["8+", "0+"],
        }, index=["A", "B"])
        sample = qualitative_samples(outcomes, catalog, sample_size=1)
        self.assertTrue(bool(sample.iloc[0]["age_mismatch"]))
        self.assertTrue(bool(sample.iloc[0]["category_mismatch"]))

    @staticmethod
    def _transfer_fixture():
        orders = pd.DataFrame({
            "Order ID": [1, 1, 2, 2, 3, 3, 4],
            "SKU": ["S", "BOX", "S", "BOX", "S", "BOX", "R"],
        })
        catalog = pd.DataFrame({
            "Product Name": ["Rare LEGO set", "Popular LEGO set", "Storage box"],
            "Κατηγορίες προϊόντων": ["LEGO", "LEGO", "Storage"],
            "Προϊόν Ηλικία": ["6+", "6+", "6+"],
            "Προϊόν Ήρωας": ["City", "City", "Unknown"],
            "Μάρκες": ["LEGO", "LEGO", "Other"],
            "Προϊόν Φύλλο": ["Unisex", "Unisex", "Unisex"],
        }, index=["R", "S", "BOX"])
        return orders, catalog

    def test_metadata_transfer_inherits_complement(self):
        orders, catalog = self._transfer_fixture()
        model = MetadataTransferRecommender().fit(orders=orders, product_catalog=catalog)
        result = model.recommend("R", top_n=2)
        self.assertEqual(result.iloc[0]["recommended_sku"], "BOX")
        self.assertGreater(result.iloc[0]["transferred_complement_score"], 0)
        self.assertGreater(result.iloc[0]["metadata_bridges"], 0)

    def test_metadata_enhanced_engine_handles_isolated_product(self):
        orders, catalog = self._transfer_fixture()
        vector = Product2VecRecommender(
            n_components=8, walk_length=4, walks_per_node=1, epochs=1, random_state=3
        ).fit(orders, catalog)
        adamic = AdamicAdarRecommender().fit(graph=vector.graph)
        base = RecommendationEngine(vector, adamic, catalog)
        transfer = MetadataTransferRecommender().fit(graph=vector.graph, product_catalog=catalog)
        result = MetadataEnhancedEngine(base, transfer).recommend(["R"], top_n=1)
        self.assertEqual(result.iloc[0]["recommended_sku"], "BOX")
        self.assertGreater(result.iloc[0]["transferred_complement_score"], 0)

    def test_multisignal_hybrid_and_metadata_score(self):
        orders, transfer_catalog = self._transfer_fixture()
        catalog = pd.DataFrame(index=transfer_catalog.index)
        for field in DEFAULT_METADATA_WEIGHTS:
            catalog[field] = transfer_catalog[field]
        vector = Product2VecRecommender(
            n_components=8, walk_length=4, walks_per_node=1, epochs=1,
            random_state=3,
        ).fit(orders, catalog)
        copurchase = CoPurchaseRecommender().fit(orders, catalog)
        adamic = AdamicAdarRecommender().fit(graph=copurchase.graph)
        metadata = MetadataSimilarityRecommender().fit(catalog)
        text = TfidfNameRecommender().fit(transfer_catalog)
        engine = HybridRecommendationEngine(
            copurchase, vector, adamic, catalog,
            weights={"copurchase": 0.3, "product2vec": 0.3,
                     "metadata": 0.2, "text": 0.2},
            metadata_model=metadata, text_model=text,
        )
        result = engine.recommend(["R"], top_n=2)
        self.assertFalse(result.empty)
        self.assertIn("copurchase_score", result.columns)
        self.assertIn("metadata_score", result.columns)
        self.assertIn("text_score", result.columns)
        self.assertAlmostEqual(sum(engine.weights.values()), 1.0)

    def test_heterogeneous_graph_excludes_unknown_and_downweights_hubs(self):
        orders, catalog = self._transfer_fixture()
        graph = build_copurchase_graph(orders)
        heterogeneous = build_heterogeneous_graph(graph, catalog)
        self.assertIn("product::R", heterogeneous)
        self.assertFalse(any(node.endswith("::unknown") for node in heterogeneous))
        self.assertTrue(any(data.get("node_type") == "category"
                            for _, data in heterogeneous.nodes(data=True)))

    def test_bootstrap_intervals_are_reproducible(self):
        outcomes = pd.DataFrame({
            "hit_at_k": [0, 1, 1, 0],
            "reciprocal_rank_at_k": [0.0, 1.0, 0.5, 0.0],
        })
        first = bootstrap_ranking_intervals(outcomes, n_bootstrap=100, random_state=7)
        second = bootstrap_ranking_intervals(outcomes, n_bootstrap=100, random_state=7)
        pd.testing.assert_frame_equal(first, second)

    def test_graph_visualization_selection_and_statistics(self):
        graph = build_copurchase_graph(self.orders)
        statistics = graph_statistics(graph).set_index("statistic")["value"]
        self.assertEqual(statistics["nodes"], 3)
        self.assertEqual(statistics["edges"], 3)
        self.assertLessEqual(strongest_edge_backbone(graph, max_edges=2).number_of_edges(), 2)
        neighborhood, levels = neighborhood_subgraph(graph, "A", direct_limit=1,
                                                      second_hop_limit=1)
        self.assertIn("A", neighborhood)
        self.assertEqual(levels["A"], 0)


if __name__ == "__main__":
    unittest.main()
