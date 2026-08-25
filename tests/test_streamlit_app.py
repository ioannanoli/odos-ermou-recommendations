"""Tests for the Streamlit interface's data preparation helpers."""

import unittest

import pandas as pd

import streamlit_app


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame(
            {
                "Product Name": ["Building Set", "Storage Box"],
                "Μάρκες": ["Example", "Example"],
                "Προϊόν Ηλικία": ["6+", "Unknown"],
            },
            index=pd.Index(["0012", "BOX"], name="SKU"),
        )

    def test_product_label_is_searchable_by_sku_and_name(self):
        self.assertEqual(
            streamlit_app.product_label("0012", self.catalog),
            "0012 — Building Set",
        )

    def test_metadata_values_exclude_unknown(self):
        self.assertEqual(
            streamlit_app.metadata_values(self.catalog, "Προϊόν Ηλικία"), ["6+"]
        )

    def test_inventory_csv_preserves_leading_zeroes(self):
        inventory = streamlit_app.read_inventory_file(
            "inventory.csv", b"SKU,Stock\n0012,3\nBOX,1\n"
        )
        self.assertEqual(inventory["SKU"].tolist(), ["0012", "BOX"])

    def test_prepare_results_adds_rank(self):
        source = pd.DataFrame(
            {
                "recommended_sku": ["BOX"],
                "Product Name": ["Storage Box"],
                "recommendation_score": [0.8],
            }
        )
        result = streamlit_app.prepare_results(source)
        self.assertEqual(result.iloc[0]["Rank"], 1)
        self.assertEqual(result.iloc[0]["recommended_sku"], "BOX")

    def test_combined_results_keep_every_active_score(self):
        source = pd.DataFrame({
            "recommended_sku": ["BOX"],
            "recommendation_score": [0.8],
            "copurchase_score": [1.0],
            "product2vec_score": [0.7],
            "metadata_score": [0.6],
            "text_score": [0.5],
        })
        result = streamlit_app.prepare_results(source)
        self.assertTrue({
            "recommendation_score", "copurchase_score", "product2vec_score",
            "metadata_score",
            "text_score",
        }.issubset(result.columns))

    def test_score_chart_contains_components_but_not_final_score(self):
        self.assertEqual(
            set(streamlit_app.COMBINED_SCORE_FIELDS.values()),
            {"copurchase_score", "product2vec_score", "metadata_score",
             "text_score"},
        )

    def test_score_formula_uses_active_weights(self):
        formula = streamlit_app.score_formula({
            "copurchase": 0.3, "product2vec": 0.2,
            "metadata": 0.2, "text": 0.3, "adamic_adar": 0.0,
        })
        self.assertIn("30% TF-IDF product-name similarity", formula)
        self.assertNotIn("Adamic", formula)

    def test_query_signature_changes_with_inventory_and_filters(self):
        base = streamlit_app.query_signature("0012", 10, {}, None)
        filtered = streamlit_app.query_signature(
            "0012", 10, {"Μάρκες": ["Example"]}, {"BOX"}
        )
        self.assertNotEqual(base, filtered)


if __name__ == "__main__":
    unittest.main()
