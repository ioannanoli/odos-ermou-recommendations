"""Tests for Phase 7 package metadata and analytical report structure."""

from pathlib import Path
import tomllib
import unittest

import main
import metadata_transfer_experiment
import graph_visualizations
import experiment_runner
import phase4_experiments
import phase6_analysis
import serve_recommendations
import streamlit_app
import visualization


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_console_entry_points_resolve(self):
        self.assertTrue(callable(main.main))
        self.assertTrue(callable(phase4_experiments.main))
        self.assertTrue(callable(phase6_analysis.main))
        self.assertTrue(callable(metadata_transfer_experiment.main))
        self.assertTrue(callable(graph_visualizations.main))
        self.assertTrue(callable(experiment_runner.main))
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
             "odos-improvement-plots"},
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
        }
        self.assertEqual({path.name for path in (ROOT / "report").glob("*.md")}, expected)


if __name__ == "__main__":
    unittest.main()
