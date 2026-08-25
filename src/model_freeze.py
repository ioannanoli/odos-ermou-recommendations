"""Integrity and tuning guards for the frozen product-page model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HISTORICAL_TEST_CONFIG_PATH = Path(
    "outputs/improvement_experiments/final/best_dev_configuration.json"
)
DEFAULT_CONFIG_PATH = Path(
    "outputs/single_sku_evaluation/development/best_product_page_configuration.json"
)
FREEZE_MANIFEST_PATH = Path(
    "outputs/single_sku_evaluation/development/model_freeze_manifest.json"
)


def sha256_file(path) -> str:
    """Return the lowercase SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_freeze_manifest(path=FREEZE_MANIFEST_PATH):
    """Load and minimally validate the model freeze record."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model freeze manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status", "configuration_path", "configuration_sha256",
        "serving_training_cutoff", "tuning_closed",
    }
    missing = required.difference(manifest)
    if missing:
        raise ValueError(f"Freeze manifest is missing keys: {sorted(missing)}")
    if manifest["status"] != "frozen":
        raise ValueError("The model manifest does not have frozen status.")
    return manifest


def verify_frozen_configuration(config_path=DEFAULT_CONFIG_PATH,
                                manifest_path=FREEZE_MANIFEST_PATH):
    """Reject any byte-level change to the configuration named by the manifest."""
    config_path = Path(config_path)
    manifest = load_freeze_manifest(manifest_path)
    locked_path = Path(manifest["configuration_path"])
    if config_path.resolve() != locked_path.resolve():
        return manifest
    actual = sha256_file(config_path)
    expected = str(manifest["configuration_sha256"]).casefold()
    if actual != expected:
        raise RuntimeError(
            "Frozen configuration integrity check failed. Restore the locked "
            "configuration instead of changing weights on the closed development set."
        )
    return manifest


def verify_frozen_artifacts(manifest_path=FREEZE_MANIFEST_PATH):
    """Verify the configuration, query set, and archived weight search hashes."""
    manifest = load_freeze_manifest(manifest_path)
    pairs = (
        ("configuration_path", "configuration_sha256"),
        ("development_queries_path", "development_queries_sha256"),
        ("weight_search_path", "weight_search_sha256"),
    )
    for path_key, hash_key in pairs:
        if path_key not in manifest or hash_key not in manifest:
            raise ValueError(f"Freeze manifest is missing {path_key!r} or {hash_key!r}.")
        path = Path(manifest[path_key])
        if not path.exists() or sha256_file(path) != str(manifest[hash_key]).casefold():
            raise RuntimeError(f"Frozen artifact integrity check failed: {path}")
    return manifest


def assert_development_tuning_open(manifest_path=FREEZE_MANIFEST_PATH):
    """Raise when a caller attempts to tune after the model has been frozen."""
    manifest = load_freeze_manifest(manifest_path)
    if manifest.get("tuning_closed", False):
        raise RuntimeError(
            "Development tuning is closed for the frozen 76-query protocol. "
            "Evaluate the unchanged model on orders strictly after "
            f"{manifest['serving_training_cutoff']} instead."
        )
    return manifest


def assert_legacy_test_reuse_allowed(manifest_path=FREEZE_MANIFEST_PATH):
    """Prevent the new candidate from being assessed on the already-seen old test."""
    manifest = load_freeze_manifest(manifest_path)
    if not manifest.get("historical_test_reopened", False):
        raise RuntimeError(
            "The historical test period is closed and must not be reopened for "
            "the TF-IDF candidate. Use future_period_evaluation.py with orders "
            f"strictly after {manifest['serving_training_cutoff']}."
        )
    return manifest
