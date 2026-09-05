"""Leakage-safe recency weights for historical order baskets."""

from __future__ import annotations

import numpy as np
import pandas as pd


DAYS_PER_MONTH = 365.25 / 12.0


def order_time_weights(orders: pd.DataFrame, half_life_months: float | None = None,
                       reference_date=None) -> pd.Series:
    """Return one exponential-decay weight per order.

    The default reference is the latest date in the supplied training frame,
    never a date from a later evaluation split.
    """
    required = {"Order ID", "Order Date"}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Time weighting requires columns: {sorted(missing)}")
    dates = pd.to_datetime(orders["Order Date"], errors="raise")
    order_dates = dates.groupby(orders["Order ID"]).min()
    if half_life_months is None:
        return pd.Series(1.0, index=order_dates.index, name="time_weight")
    if half_life_months <= 0:
        raise ValueError("half_life_months must be positive or None.")
    reference = pd.Timestamp(reference_date) if reference_date is not None else order_dates.max()
    if reference < order_dates.max():
        raise ValueError("reference_date cannot precede an order in the training data.")
    age_months = (reference - order_dates).dt.total_seconds() / (86400.0 * DAYS_PER_MONTH)
    weights = np.power(0.5, age_months / float(half_life_months))
    return pd.Series(weights, index=order_dates.index, name="time_weight")
