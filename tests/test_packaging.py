"""Tests for Phase 7 package metadata and analytical report structure."""

import csv
import json
from pathlib import Path
import tomllib
import unittest

import main
from experiments import (candidate_health_report, expanded_candidate_generation,
                         experiment_runner, future_period_evaluation,
                         graph_visualizations, logistic_ranker_experiment,
                         metadata_transfer_experiment, phase4_experiments,
                         phase6_analysis, single_sku_evaluation, visualization)
import serve_recommendations
import streamlit_app
from src.model_freeze import sha256_file


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_console_entry_points_resolve(self):
        self.assertTrue(callable(main.main))
        self.assertTrue(callable(phase4_experiments.main))
        self.assertTrue(callable(phase6_analysis.main))
        self.assertTrue(callable(single_sku_evaluation.main))
        self.assertTrue(callable(metadata_transfer_experiment.main))
        self.assertTrue(callable(graph_visualizations.main))
        self.assertTrue(callable(experiment_runner.main))
        self.assertTrue(callable(future_period_evaluation.main))
        self.assertTrue(callable(expanded_candidate_generation.main))
        self.assertTrue(callable(candidate_health_report.main))
        self.assertTrue(callable(logistic_ranker_experiment.main))
        self.assertTrue(callable(visualization.main))
        self.assertTrue(callable(serve_recommendations.main))
        self.assertTrue(callable(streamlit_app.launch))

    def test_pyproject_metadata_and_scripts(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["name"], "odos-ermou-recommendations")
        self.assertEqual(
            set(project["scripts"]),
            {"odos-recommend", "odos-app", "odos-pipeline", "odos-tune", "odos-analyze",
             "odos-metadata-transfer", "odos-visualize", "odos-improve",
             "odos-improvement-plots", "odos-evaluate-product-page",
             "odos-evaluate-future", "odos-expanded-candidates",
             "odos-candidate-health", "odos-logistic-backtest"},
        )

    def test_analytical_report_has_all_sections(self):
        expected = {
            "README.md",
            "01_introduction.md",
            "02_data_collection.md",
            "03_data_processing.md",
            "04_methodology_and_algorithms.md",
            "05_experimental_setup.md",
            "06_results.md",
            "07_discussion.md",
            "FULL_REPORT.md",
        }
        self.assertEqual({path.name for path in (ROOT / "report").glob("*.md")}, expected)

    def test_experiments_are_isolated_from_production_entry_points(self):
        experiment_modules = {
            "phase4_experiments.py", "phase6_analysis.py",
            "single_sku_evaluation.py", "future_period_evaluation.py",
            "expanded_candidate_generation.py", "candidate_health_report.py",
            "logistic_ranker_experiment.py", "metadata_transfer_experiment.py",
            "graph_visualizations.py", "experiment_runner.py", "visualization.py",
        }
        self.assertTrue(all((ROOT / "experiments" / name).exists()
                            for name in experiment_modules))
        self.assertTrue(all(not (ROOT / name).exists()
                            for name in experiment_modules))

    def test_final_weight_files_match_model_configurations(self):
        config_directory = ROOT / "model_configs"
        self.assertTrue((config_directory / "final_model_hyperparameters.csv").exists())
        product_page = json.loads(
            (config_directory / "final_product_page_model.json").read_text(
                encoding="utf-8"
            )
        )
        historical = json.loads(
            (config_directory / "historical_basket_model.json").read_text(
                encoding="utf-8"
            )
        )

        def weight_group(file_name, group):
            with (config_directory / file_name).open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = csv.DictReader(stream)
                return {
                    row["component"]: float(row["weight"])
                    for row in rows if row["weight_group"] == group
                }

        self.assertEqual(
            weight_group("final_product_page_weights.csv", "final_blend"),
            product_page["blend_weights"],
        )
        self.assertEqual(
            weight_group("historical_basket_weights.csv", "final_blend"),
            historical["blend_weights"],
        )
        self.assertEqual(
            weight_group("final_product_page_weights.csv", "heterogeneous_graph"),
            product_page["heterogeneous_relationship_weights"],
        )
        self.assertEqual(
            weight_group("historical_basket_weights.csv", "heterogeneous_graph"),
            historical["heterogeneous_relationship_weights"],
        )
        self.assertEqual(
            weight_group("final_product_page_weights.csv", "tfidf_channel"),
            {
                "word": product_page["tfidf"]["word_weight"],
                "character": product_page["tfidf"]["char_weight"],
            },
        )
        self.assertEqual(
            sorted(weight_group(
                "final_product_page_weights.csv", "metadata_field"
            ).values()),
            sorted(product_page["metadata_field_weights"].values()),
        )

    def test_learned_weight_exports_match_manifest(self):
        export_directory = ROOT / "model_weights"
        manifest = json.loads(
            (export_directory / "export_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["graph"]["edges"], 8841)
        self.assertEqual(manifest["node2vec"]["dimensions"], 48)
        for file_name, details in manifest["files"].items():
            self.assertEqual(sha256_file(export_directory / file_name),
                             details["sha256"])


if __name__ == "__main__":
    unittest.main()
