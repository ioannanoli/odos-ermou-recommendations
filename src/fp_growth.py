"""Small dependency-free FP-Growth implementation for transaction baskets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil

import pandas as pd


@dataclass
class _Node:
    item: str | None
    count: int = 0
    parent: "_Node | None" = None
    children: dict[str, "_Node"] = field(default_factory=dict)
    link: "_Node | None" = None


def _build_tree(weighted_transactions, min_count):
    frequencies = Counter()
    for transaction, weight in weighted_transactions:
        for item in set(transaction):
            frequencies[item] += weight
    frequent = {item: count for item, count in frequencies.items() if count >= min_count}
    if not frequent:
        return None, {}, frequent

    root, heads, tails = _Node(None), {}, {}
    for transaction, weight in weighted_transactions:
        items = sorted(
            (item for item in set(transaction) if item in frequent),
            key=lambda item: (-frequent[item], item),
        )
        node = root
        for item in items:
            child = node.children.get(item)
            if child is None:
                child = node.children[item] = _Node(item, parent=node)
                if item in tails:
                    tails[item].link = child
                else:
                    heads[item] = child
                tails[item] = child
            child.count += weight
            node = child
    return root, heads, frequent


def _mine(weighted_transactions, min_count, suffix=frozenset(), max_length=None):
    _, heads, frequencies = _build_tree(weighted_transactions, min_count)
    for item in sorted(frequencies, key=lambda value: (frequencies[value], value)):
        itemset = suffix | {item}
        if max_length is None or len(itemset) <= max_length:
            yield itemset, frequencies[item]
        paths = []
        node = heads[item]
        while node is not None:
            path, parent = [], node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                paths.append((path, node.count))
            node = node.link
        if max_length is None or len(itemset) < max_length:
            yield from _mine(paths, min_count, itemset, max_length)


def frequent_itemsets(orders: pd.DataFrame, min_support=0.01, max_length=None) -> pd.DataFrame:
    """Extract frequent SKU sets from baskets using FP-Growth."""
    baskets = orders.groupby("Order ID")["SKU"].apply(
        lambda values: tuple(set(values.dropna().astype(str)))
    )
    n_orders = len(baskets)
    columns = ["itemset", "length", "count", "support"]
    if n_orders == 0:
        return pd.DataFrame(columns=columns)
    min_count = ceil(min_support * n_orders) if isinstance(min_support, float) else int(min_support)
    if min_count < 1:
        raise ValueError("min_support must imply at least one order.")
    found = {
        tuple(sorted(itemset)): count
        for itemset, count in _mine(
            [(basket, 1) for basket in baskets], min_count, max_length=max_length
        )
    }
    rows = [
        {"itemset": itemset, "length": len(itemset), "count": count, "support": count / n_orders}
        for itemset, count in found.items()
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["support", "length"], ascending=[False, True], ignore_index=True
    )
