"""Interactive Streamlit interface for the frozen Odos Ermou recommender."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

from serve_recommendations import MODEL_PATH
from src.final_recommender import FinalRecommender


PRODUCT_NAME = "Product Name"
CATEGORY = "Κατηγορίες προϊόντων"
BRAND = "Μάρκες"
AGE = "Προϊόν Ηλικία"
HERO = "Προϊόν Ήρωας"
GENDER = "Προϊόν Φύλλο"

FILTER_FIELDS = {
    "Category": CATEGORY,
    "Brand": BRAND,
    "Age": AGE,
    "Hero": HERO,
    "Gender": GENDER,
}
DISPLAY_FIELDS = [PRODUCT_NAME, CATEGORY, BRAND, AGE, HERO, GENDER]
SCORE_FIELDS = {
    "Co-purchase": "copurchase_score",
    "Product2Vec": "product2vec_score",
    "Metadata": "metadata_score",
    "Final score": "recommendation_score",
}


@st.cache_resource(show_spinner="Loading the trained recommendation model…")
def load_model(model_path: str, modified_ns: int) -> FinalRecommender:
    """Load and cache a trusted model, refreshing after its file changes."""
    del modified_ns
    return FinalRecommender.load(model_path)


@st.cache_data(show_spinner=False)
def read_inventory_file(file_name: str, content: bytes) -> pd.DataFrame:
    """Read an uploaded inventory workbook while preserving SKU text."""
    suffix = Path(file_name).suffix.casefold()
    buffer = BytesIO(content)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(buffer, dtype="string")
    raise ValueError("Inventory must be a CSV, XLSX, or XLS file.")


def product_label(sku: str, catalog: pd.DataFrame) -> str:
    """Create the searchable SKU and product-name label used by the picker."""
    name = catalog.at[sku, PRODUCT_NAME] if PRODUCT_NAME in catalog else ""
    name = "" if pd.isna(name) else str(name).strip()
    return f"{sku} — {name}" if name else str(sku)


def metadata_values(catalog: pd.DataFrame, field: str) -> list[str]:
    """Return sorted, usable catalog values for one UI filter."""
    if field not in catalog:
        return []
    values = {
        str(value).strip()
        for value in catalog[field].dropna()
        if str(value).strip() and str(value).strip().casefold() != "unknown"
    }
    return sorted(values, key=str.casefold)


def prepare_results(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Add rank and select the user-facing recommendation columns."""
    if recommendations.empty:
        return recommendations.copy()
    result = recommendations.copy()
    result.insert(0, "Rank", range(1, len(result) + 1))
    ordered = [
        "Rank", "recommended_sku", PRODUCT_NAME, CATEGORY, BRAND, AGE, HERO,
        GENDER, "recommendation_score", "copurchase_score",
        "product2vec_score", "metadata_score",
    ]
    return result[[column for column in ordered if column in result]]


def query_signature(cart_skus, top_n, filters, available_skus):
    """Describe every input that affects a recommendation result."""
    filter_signature = tuple(
        (field, tuple(values)) for field, values in sorted(filters.items())
    )
    inventory_signature = (
        None if available_skus is None else frozenset(available_skus)
    )
    return tuple(cart_skus), int(top_n), filter_signature, inventory_signature


def _model_sidebar(model: FinalRecommender) -> None:
    st.sidebar.header("Model")
    summary = model.training_summary
    st.sidebar.metric("Catalog products", f"{summary.get('catalog_skus', 0):,}")
    st.sidebar.metric("Historical orders", f"{summary.get('order_count', 0):,}")
    st.sidebar.metric("Graph connections", f"{summary.get('graph_edges', 0):,}")
    training_end = str(summary.get("training_end", "Unknown")).split()[0]
    st.sidebar.caption(f"Training data through {training_end}")


def _inventory_sidebar() -> set[str] | None:
    st.sidebar.header("Inventory (optional)")
    uploaded = st.sidebar.file_uploader(
        "Upload available products",
        type=["csv", "xlsx", "xls"],
        help="Only products listed in the selected SKU column will be recommended.",
    )
    if uploaded is None:
        st.sidebar.caption("No inventory restriction is active.")
        return None
    try:
        inventory = read_inventory_file(uploaded.name, uploaded.getvalue())
    except Exception as error:
        st.sidebar.error(f"Could not read inventory: {error}")
        return set()
    if inventory.empty or len(inventory.columns) == 0:
        st.sidebar.warning("The uploaded inventory is empty.")
        return set()
    default = inventory.columns.get_loc("SKU") if "SKU" in inventory else 0
    column = st.sidebar.selectbox(
        "Inventory SKU column", inventory.columns.tolist(), index=int(default)
    )
    available = set(inventory[column].dropna().astype(str).str.strip()) - {""}
    st.sidebar.success(f"Inventory filter: {len(available):,} SKUs")
    return available


def _metadata_filters(catalog: pd.DataFrame) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    with st.expander("Optional metadata filters"):
        st.caption(
            "A result must match every filter you select. Leave all fields empty "
            "to use the model's normal ranking."
        )
        columns = st.columns(2)
        for position, (label, field) in enumerate(FILTER_FIELDS.items()):
            selected = columns[position % 2].multiselect(
                label, metadata_values(catalog, field), key=f"filter_{field}"
            )
            if selected:
                filters[field] = selected
    return filters


def _selected_product_details(catalog: pd.DataFrame, cart_skus: list[str]) -> None:
    if not cart_skus:
        return
    with st.expander("Selected product details", expanded=False):
        details = catalog.loc[cart_skus, [
            field for field in DISPLAY_FIELDS if field in catalog
        ]].reset_index(names="SKU")
        st.dataframe(details, hide_index=True, width="stretch")


def _show_results(results: pd.DataFrame, cart_skus: list[str]) -> None:
    if results.empty:
        st.warning(
            "No recommendations matched this cart and the active filters. "
            "Try removing an inventory or metadata restriction."
        )
        return
    st.subheader(f"Top {len(results)} recommendations")
    table_tab, scores_tab = st.tabs(["Products", "Why these products?"])
    with table_tab:
        st.dataframe(
            results,
            hide_index=True,
            width="stretch",
            column_config={
                "recommended_sku": st.column_config.TextColumn("SKU"),
                PRODUCT_NAME: st.column_config.TextColumn("Product", width="large"),
                CATEGORY: st.column_config.TextColumn("Category", width="large"),
                BRAND: st.column_config.TextColumn("Brand"),
                AGE: st.column_config.TextColumn("Age"),
                HERO: st.column_config.TextColumn("Hero"),
                GENDER: st.column_config.TextColumn("Gender"),
                "recommendation_score": st.column_config.NumberColumn(
                    "Final score", format="%.3f"
                ),
                "copurchase_score": st.column_config.NumberColumn(
                    "Co-purchase", format="%.3f"
                ),
                "product2vec_score": st.column_config.NumberColumn(
                    "Product2Vec", format="%.3f"
                ),
                "metadata_score": st.column_config.NumberColumn(
                    "Metadata", format="%.3f"
                ),
            },
        )
    with scores_tab:
        st.caption(
            "The final ranking blends co-purchase evidence (40%), Product2Vec "
            "similarity (30%), and metadata similarity (30%)."
        )
        score_columns = {
            label: field for label, field in SCORE_FIELDS.items() if field in results
        }
        chart = results.set_index("recommended_sku")[list(score_columns.values())]
        chart = chart.rename(columns={value: key for key, value in score_columns.items()})
        st.bar_chart(chart)
    csv = results.to_csv(index=False).encode("utf-8-sig")
    cart_name = "_".join(cart_skus[:3])
    st.download_button(
        "Download recommendations as CSV",
        data=csv,
        file_name=f"recommendations_{cart_name}.csv",
        mime="text/csv",
    )


def main() -> None:
    """Render the local recommendation interface."""
    st.set_page_config(
        page_title="Odos Ermou Recommendations",
        page_icon="🧸",
        layout="wide",
    )
    st.title("🧸 Odos Ermou Product Recommendations")
    st.write(
        "Search for one or more products to build a cart, then generate products "
        "that customers are likely to buy with them."
    )

    model_path = MODEL_PATH
    if not model_path.exists():
        st.error(f"The trained model was not found at `{model_path}`.")
        st.code("python serve_recommendations.py train", language="powershell")
        st.stop()
    try:
        model = load_model(str(model_path), model_path.stat().st_mtime_ns)
    except Exception as error:
        st.error(f"The trained model could not be loaded: {error}")
        st.stop()

    catalog = model.product_catalog.copy()
    catalog.index = catalog.index.astype(str)
    catalog.index.name = "SKU"
    _model_sidebar(model)
    available_skus = _inventory_sidebar()

    top_n = st.slider("Number of recommendations", min_value=1, max_value=30, value=10)
    cart_skus = st.multiselect(
        "Search by SKU or product name",
        options=catalog.index.tolist(),
        format_func=lambda sku: product_label(sku, catalog),
        placeholder="Type a SKU or product name…",
        help="You can select several products to represent a customer's cart.",
    )
    _selected_product_details(catalog, cart_skus)
    filters = _metadata_filters(catalog)
    current_signature = query_signature(cart_skus, top_n, filters, available_skus)

    if st.button(
        "Generate recommendations", type="primary", disabled=not cart_skus,
        width="stretch",
    ):
        try:
            with st.spinner("Ranking products…"):
                raw = model.recommend(
                    cart_skus,
                    top_n=top_n,
                    available_skus=available_skus,
                    metadata_filters=filters,
                    enrich=True,
                )
            st.session_state["recommendation_results"] = prepare_results(raw)
            st.session_state["recommendation_cart"] = cart_skus
            st.session_state["recommendation_signature"] = current_signature
        except Exception as error:
            st.error(f"Recommendations could not be generated: {error}")

    if "recommendation_results" in st.session_state:
        previous_cart = st.session_state.get("recommendation_cart", [])
        if st.session_state.get("recommendation_signature") != current_signature:
            st.info("The inputs changed. Click Generate recommendations to refresh the results.")
        _show_results(st.session_state["recommendation_results"], previous_cart)


def launch() -> None:
    """Launch this module through the installed ``odos-app`` command."""
    from streamlit.web import cli as streamlit_cli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
