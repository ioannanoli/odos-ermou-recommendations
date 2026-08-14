"""Validated model settings shared by production and analysis pipelines."""

SELECTED_PRODUCT2VEC_CONFIG = {
    "n_components": 48,
    "walk_length": 8,
    "walks_per_node": 2,
    "window_size": 3,
    "negative_samples": 2,
    "epochs": 2,
    "learning_rate": 0.05,
    "p": 1.0,
    "q": 1.0,
}
