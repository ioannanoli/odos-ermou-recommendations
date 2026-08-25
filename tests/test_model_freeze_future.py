import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from future_period_evaluation import _ensure_empty_output, strictly_future_orders
from src.model_freeze import (DEFAULT_CONFIG_PATH, FREEZE_MANIFEST_PATH,
                              assert_development_tuning_open,
                              assert_legacy_test_reuse_allowed,
                              sha256_file, verify_frozen_artifacts,
                              verify_frozen_configuration)


class ModelFreezeAndFutureEvaluationTests(unittest.TestCase):
    def test_repository_configuration_matches_freeze_hash(self):
        manifest = verify_frozen_configuration(
            DEFAULT_CONFIG_PATH, FREEZE_MANIFEST_PATH
        )
        self.assertTrue(manifest["tuning_closed"])
        self.assertEqual(manifest["development_query_count"], 76)
        verify_frozen_artifacts(FREEZE_MANIFEST_PATH)

    def test_modified_frozen_configuration_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            manifest_path = root / "manifest.json"
            config.write_text('{"weight": 0.4}', encoding="utf-8")
            manifest_path.write_text(json.dumps({
                "status": "frozen",
                "configuration_path": str(config),
                "configuration_sha256": sha256_file(config),
                "serving_training_cutoff": "2026-06-19T14:28:25",
                "tuning_closed": True,
            }), encoding="utf-8")
            verify_frozen_configuration(config, manifest_path)
            config.write_text('{"weight": 0.5}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                verify_frozen_configuration(config, manifest_path)

    def test_development_tuning_is_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Development tuning is closed"):
            assert_development_tuning_open()

    def test_historical_test_period_is_closed(self):
        with self.assertRaisesRegex(RuntimeError, "historical test period is closed"):
            assert_legacy_test_reuse_allowed()

    def test_future_rows_must_be_strictly_after_cutoff(self):
        orders = pd.DataFrame({
            "Order ID": [1, 2, 3],
            "Order Date": ["2026-06-19 14:28:25", "2026-06-19 14:28:26",
                           "2026-07-01"],
            "SKU": ["A", "B", "C"],
        })
        result = strictly_future_orders(orders, "2026-06-19 14:28:25")
        self.assertEqual(result["Order ID"].tolist(), [2, 3])

    def test_future_output_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.assertEqual(_ensure_empty_output(output), output)
            (output / "summary_metrics.csv").write_text("metric,value\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _ensure_empty_output(output)


if __name__ == "__main__":
    unittest.main()
