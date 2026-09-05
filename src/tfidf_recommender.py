"""Dependency-free TF-IDF similarity for product-name recommendations."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log, sqrt
import re
import unicodedata

import pandas as pd


PRODUCT_NAME_COLUMN = "Product Name"


def normalize_product_name(value) -> str:
    """Normalize a multilingual product name while retaining letters and digits."""
    if pd.isna(value) or str(value).strip().casefold() == "unknown":
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _ngrams(values, lower, upper):
    for size in range(lower, upper + 1):
        for index in range(len(values) - size + 1):
            yield values[index:index + size]


class TfidfNameRecommender:
    """Rank products by word and character TF-IDF cosine similarity."""

    def __init__(self, word_weight=0.6, char_weight=0.4,
                 word_ngram_range=(1, 2), char_ngram_range=(3, 5)):
        if word_weight < 0 or char_weight < 0 or word_weight + char_weight <= 0:
            raise ValueError("TF-IDF channel weights must be non-negative and non-zero.")
        for name, value in (("word_ngram_range", word_ngram_range),
                            ("char_ngram_range", char_ngram_range)):
            if len(value) != 2 or value[0] < 1 or value[1] < value[0]:
                raise ValueError(f"{name} must be an increasing pair of positive integers.")
        total = word_weight + char_weight
        self.channel_weights = {"word": word_weight / total, "char": char_weight / total}
        self.word_ngram_range = tuple(int(value) for value in word_ngram_range)
        self.char_ngram_range = tuple(int(value) for value in char_ngram_range)
        self.product_catalog = None
        self._vectors = {"word": {}, "char": {}}
        self._indexes = {"word": defaultdict(list), "char": defaultdict(list)}
        self._cart_cache = {}

    def _channel_counts(self, text, channel):
        if not text:
            return Counter()
        if channel == "word":
            tokens = text.split()
            features = (" ".join(parts) for parts in _ngrams(
                tokens, *self.word_ngram_range
            ))
        else:
            compact = f"^{text}$"
            features = _ngrams(compact, *self.char_ngram_range)
        return Counter(features)

    def _fit_channel(self, texts, channel):
        counts_by_sku = {
            sku: self._channel_counts(text, channel) for sku, text in texts.items()
        }
        document_frequency = Counter()
        for counts in counts_by_sku.values():
            document_frequency.update(counts.keys())
        document_count = len(counts_by_sku)
        vectors = {}
        index = defaultdict(list)
        for sku, counts in counts_by_sku.items():
            weighted = {
                feature: (1.0 + log(count)) *
                         (log((1.0 + document_count) /
                              (1.0 + document_frequency[feature])) + 1.0)
                for feature, count in counts.items()
            }
            norm = sqrt(sum(value * value for value in weighted.values()))
            vector = ({feature: value / norm for feature, value in weighted.items()}
                      if norm else {})
            vectors[sku] = vector
            for feature, value in vector.items():
                index[feature].append((sku, value))
        self._vectors[channel] = vectors
        self._indexes[channel] = index

    def fit(self, product_catalog):
        """Build sparse word and character TF-IDF indexes from product names."""
        if PRODUCT_NAME_COLUMN not in product_catalog:
            raise ValueError(f"Product catalog is missing {PRODUCT_NAME_COLUMN!r}.")
        self.product_catalog = product_catalog.copy()
        texts = {
            str(sku): normalize_product_name(value)
            for sku, value in product_catalog[PRODUCT_NAME_COLUMN].items()
        }
        for channel in self.channel_weights:
            self._fit_channel(texts, channel)
        self._cart_cache.clear()
        return self

    def similarity(self, left_sku, right_sku):
        """Return combined word/character cosine similarity for two SKUs."""
        left_sku, right_sku = str(left_sku), str(right_sku)
        score = 0.0
        for channel, channel_weight in self.channel_weights.items():
            left = self._vectors[channel].get(left_sku, {})
            right = self._vectors[channel].get(right_sku, {})
            if len(left) > len(right):
                left, right = right, left
            score += channel_weight * sum(
                value * right.get(feature, 0.0) for feature, value in left.items()
            )
        return float(score)

    def _recommend_one(self, sku):
        scores = defaultdict(float)
        for channel, channel_weight in self.channel_weights.items():
            for feature, query_weight in self._vectors[channel].get(sku, {}).items():
                for candidate, candidate_weight in self._indexes[channel].get(feature, ()):
                    if candidate != sku:
                        scores[candidate] += (
                            channel_weight * query_weight * candidate_weight
                        )
        return scores

    def recommend_cart(self, cart_skus, top_n=10):
        """Rank catalog products by maximum name similarity to any query SKU."""
        if top_n < 1:
            raise ValueError("top_n must be positive.")
        cart = list(dict.fromkeys(str(sku) for sku in cart_skus))
        cache_key = (tuple(cart), int(top_n))
        if cache_key in self._cart_cache:
            return self._cart_cache[cache_key].copy()
        cart_set = set(cart)
        scores = defaultdict(float)
        for sku in cart:
            for candidate, score in self._recommend_one(sku).items():
                if candidate not in cart_set:
                    scores[candidate] = max(scores[candidate], score)
        rows = [{
            "cart_skus": " | ".join(cart),
            "recommended_sku": sku,
            "text_score": score,
        } for sku, score in scores.items() if score > 0]
        columns = ["cart_skus", "recommended_sku", "text_score"]
        result = (pd.DataFrame(rows, columns=columns)
                  .sort_values(["text_score", "recommended_sku"],
                               ascending=[False, True])
                  .head(top_n).reset_index(drop=True))
        self._cart_cache[cache_key] = result
        return result.copy()

    def recommend(self, sku, top_n=10):
        """Convenience wrapper for a single product-page SKU."""
        return self.recommend_cart([sku], top_n)
