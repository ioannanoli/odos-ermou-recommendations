"""Tests for Phase 7 package metadata and analytical report structure."""

from pathlib import Path
import tomllib
import unittest

import main
import phase4_experiments
import phase6_analysis


ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def test_console_entry_points_resolve(self):
        self.assertTrue(callable(main.main))
        self.assertTrue(callable(phase4_experiments.main))
        self.assertTrue(callable(phase6_analysis.main))

    def test_pyproject_metadata_and_scripts(self):
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        self.assertEqual(project["name"], "odos-ermou-recommendations")
        self.assertEqual(
            set(project["scripts"]), {"odos-recommend", "odos-tune", "odos-analyze"}
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
