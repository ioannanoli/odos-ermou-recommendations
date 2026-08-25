import unittest

import pandas as pd

from src.tfidf_recommender import (TfidfNameRecommender,
                                   normalize_product_name)


class TfidfNameRecommenderTests(unittest.TestCase):
    def setUp(self):
        self.catalog = pd.DataFrame({
            "Product Name": [
                "LEGO City Police Station 60316",
                "LEGO City Fire Station 60320",
                "Barbie Dreamhouse Doll Playset",
                "Unknown",
            ]
        }, index=["A", "B", "C", "D"])

    def test_normalization_preserves_multilingual_letters_and_digits(self):
        self.assertEqual(
            normalize_product_name("  LEGO   \u03a0\u03cc\u03bb\u03b7 60316! "),
            "lego \u03c0\u03cc\u03bb\u03b7 60316",
        )

    def test_similar_product_name_ranks_first(self):
        model = TfidfNameRecommender().fit(self.catalog)
        result = model.recommend("A", top_n=3)
        self.assertEqual(result.iloc[0]["recommended_sku"], "B")
        self.assertGreater(
            model.similarity("A", "B"), model.similarity("A", "C")
        )
        self.assertNotIn("A", result["recommended_sku"].tolist())

    def test_unknown_name_has_no_recommendations(self):
        model = TfidfNameRecommender().fit(self.catalog)
        self.assertTrue(model.recommend("D").empty)


if __name__ == "__main__":
    unittest.main()
