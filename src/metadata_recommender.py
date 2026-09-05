"""Catalog metadata similarity as an explicit recommendation signal."""

from __future__ import annotations

from collections import defaultdict
import re

import pandas as pd


DEFAULT_METADATA_WEIGHTS = {
    "Κατηγορίες προϊόντων": 0.35,
    "Μάρκες": 0.20,
    "Προϊόν Ηλικία": 0.25,
    "Προϊόν Ήρωας": 0.15,
    "Προϊόν Φύλλο": 0.05,
}
MISSING_METADATA = {"", "unknown", "nan", "none", "null", "n/a"}


def metadata_tokens(value) -> set[str]:
    """Split a potentially multi-valued catalog attribute into clean tokens."""
    if pd.isna(value) or str(value).strip().casefold() in MISSING_METADATA:
        return set()
    return {token.strip().casefold() for token in re.split(r"[,;|/>]+", str(value))
            if token.strip() and token.strip().casefold() not in MISSING_METADATA}


class MetadataSimilarityRecommender:
    """Rank products by weighted field-wise Jaccard similarity to a cart."""

    def __init__(self, field_weights=None):
        weights = dict(field_weights or DEFAULT_METADATA_WEIGHTS)
        if not weights or any(weight < 0 for weight in weights.values()) or sum(weights.values()) <= 0:
            raise ValueError("field_weights must have non-negative values and a positive total.")
        total = sum(weights.values())
        self.field_weights = {field: weight / total for field, weight in weights.items()}
        self.product_catalog = None
        self._features = {}
        self._index = defaultdict(set)
        self._cart_cache = {}

    def fit(self, product_catalog):
        missing = set(self.field_weights).difference(product_catalog.columns)
        if missing:
            raise ValueError(f"Metadata columns are missing: {sorted(missing)}")
        self.product_catalog = product_catalog.copy()
        self._features.clear()
        self._index.clear()
        self._cart_cache.clear()
        for raw_sku, row in product_catalog.iterrows():
            sku = str(raw_sku)
            fields = {field: metadata_tokens(row[field]) for field in self.field_weights}
            self._features[sku] = fields
            for field, values in fields.items():
                for value in values:
                    self._index[(field, value)].add(sku)
        return self

    def similarity(self, left_sku, right_sku):
        left = self._features.get(str(left_sku))
        right = self._features.get(str(right_sku))
        if left is None or right is None:
            return 0.0
        score = 0.0
        for field, weight in self.field_weights.items():
            union = left[field] | right[field]
            if union:
                score += weight * len(left[field] & right[field]) / len(union)
        return float(score)

    def recommend_cart(self, cart_skus, top_n=10):
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cache_key = (tuple(cart), int(top_n))
        if cache_key in self._cart_cache:
            return self._cart_cache[cache_key].copy()
        cart_set = set(cart)
        candidates = set()
        for sku in cart:
            for field, values in self._features.get(sku, {}).items():
                for value in values:
                    candidates.update(self._index[(field, value)])
        candidates.difference_update(cart_set)
        rows = []
        for candidate in candidates:
            similarities = [self.similarity(sku, candidate) for sku in cart]
            score = max(similarities, default=0.0)
            if score > 0:
                rows.append({
                    "cart_skus": " | ".join(cart),
                    "recommended_sku": candidate,
                    "metadata_score": score,
                })
        columns = ["cart_skus", "recommended_sku", "metadata_score"]
        result = (pd.DataFrame(rows, columns=columns)
                .sort_values(["metadata_score", "recommended_sku"], ascending=[False, True])
                .head(top_n).reset_index(drop=True))
        self._cart_cache[cache_key] = result
        return result.copy()
