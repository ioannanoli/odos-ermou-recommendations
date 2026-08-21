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

    def test_query_signature_changes_with_inventory_and_filters(self):
        base = streamlit_app.query_signature(["0012"], 10, {}, None)
        filtered = streamlit_app.query_signature(
            ["0012"], 10, {"Μάρκες": ["Example"]}, {"BOX"}
        )
        self.assertNotEqual(base, filtered)


if __name__ == "__main__":
    unittest.main()
