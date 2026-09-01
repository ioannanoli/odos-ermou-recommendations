import unittest

import numpy as np
import pandas as pd

from experiments.logistic_ranker_experiment import chronological_backtest_periods
from src.logistic_ranker import FEATURE_NAMES, RegularizedLogisticRanker


class LogisticRankerTests(unittest.TestCase):
    def test_regularized_logistic_ranker_learns_positive_direction(self):
        features = np.zeros((8, len(FEATURE_NAMES)), dtype=float)
        features[:, 0] = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0]
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        ranker = RegularizedLogisticRanker(l2=0.01).fit(features, labels)
        probabilities = ranker.predict_proba(features)
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertGreater(ranker.coefficients[0], 0)
        self.assertEqual(len(ranker.coefficient_frame()), len(FEATURE_NAMES))

    def test_chronological_periods_keep_orders_intact_and_before_cutoff(self):
        rows = []
        for order_id in range(1, 13):
            for sku in (f"A{order_id}", f"B{order_id}"):
                rows.append({
                    "Order ID": order_id,
                    "Order Date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=order_id),
                    "SKU": sku,
                })
        orders = pd.DataFrame(rows)
        periods = chronological_backtest_periods(
            orders, cutoff="2020-01-12", fractions=(0.25, 0.25, 0.25, 0.25)
        )
        order_sets = [set(frame["Order ID"]) for frame in periods]
        self.assertEqual(sum(len(values) for values in order_sets), 11)
        for left_index, left in enumerate(order_sets):
            for right in order_sets[left_index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertLessEqual(
            max(pd.to_datetime(frame["Order Date"]).max() for frame in periods),
            pd.Timestamp("2020-01-12"),
        )


if __name__ == "__main__":
    unittest.main()
