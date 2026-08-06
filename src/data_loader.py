"""Data loading and normalization utilities for the recommendation pipeline."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Iterable

import pandas as pd


PRODUCT_COLUMNS = [
    "SKU",
    "Product Name",
    "Κατηγορίες προϊόντων",
    "Μάρκες",
    "Προϊόν Ηλικία",
    "Προϊόν Ήρωας",
    "Προϊόν Φύλλο",
    "sex_p",
    "hero_p",
    "age_p",
]
TEXT_PRODUCT_COLUMNS = PRODUCT_COLUMNS[1:]
MISSING_TEXT = {"", "nan", "none", "null", "n/a", "na"}


def clean_text(value) -> object:
    """Return normalized Unicode text while preserving actual missing values."""
    if pd.isna(value):
        return pd.NA
    value = html.unescape(str(value))
    value = unicodedata.normalize("NFKC", value)
    value = re.sub(r"\s+", " ", value).strip()
    return pd.NA if value.casefold() in MISSING_TEXT else value


def load_orders(path, statuses: Iterable[str] | None = None) -> pd.DataFrame:
    """Load order lines.

    By default no order status is discarded, including cancelled and pending
    orders. Pass ``statuses`` to explicitly select an allow-list.
    """
    df = pd.read_excel(path)
    required = {"Order ID", "Order Status", "SKU"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required order columns are missing: {sorted(missing)}")

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].map(clean_text)

    if statuses is not None:
        allowed = {str(status).strip() for status in statuses}
        df = df[df["Order Status"].isin(allowed)].copy()

    df = df[df["Order ID"].notna() & df["SKU"].notna()].copy()
    df["SKU"] = df["SKU"].astype(str).str.strip()
    return df[~df["SKU"].str.casefold().isin(MISSING_TEXT)].copy()


def create_product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Create one normalized, imputed metadata row per SKU."""
    missing = set(PRODUCT_COLUMNS).difference(df.columns)
    if missing:
        raise ValueError(f"Required product columns are missing: {sorted(missing)}")

    catalog = df[PRODUCT_COLUMNS].copy()
    catalog["SKU"] = catalog["SKU"].astype(str).str.strip()
    for column in TEXT_PRODUCT_COLUMNS:
        catalog[column] = catalog[column].map(clean_text)

    # Prefer the most common non-null value for repeated product metadata.
    def most_common(series: pd.Series):
        values = series.dropna()
        return values.mode().iloc[0] if not values.empty else "Unknown"

    return catalog.groupby("SKU", sort=True)[TEXT_PRODUCT_COLUMNS].agg(most_common)
