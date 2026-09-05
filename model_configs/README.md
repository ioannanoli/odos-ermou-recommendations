# Final model configurations and weights

This folder makes the selected models easy to inspect for submission.

- `final_product_page_model.json` is the content-locked frozen configuration
  used by the Streamlit product-page recommender.
- `final_product_page_weights.csv` lists its selected blend, metadata,
  heterogeneous-graph, and TF-IDF channel weights.
- `historical_basket_model.json` is the earlier final basket-completion model
  used for the one-time historical test result.
- `historical_basket_weights.csv` lists its selected scalar weights.
- `final_model_hyperparameters.csv` separates graph, Node2vec, Skip-Gram, and
  TF-IDF settings from learned parameters.
- `model_registry.json` distinguishes the serving model from the historical
  comparison model and rejected experiments.

The CSV files contain selected scoring weights. Product2Vec's learned
per-product embedding matrix is fitted from the supplied order history and is
stored inside the local `models/final_recommender.pkl` artifact. Submission
exports of the fitted graph and embedding matrix are generated under
`model_weights/`.

Do not change `final_product_page_model.json` in place. Its canonical SHA-256 hash is
checked against the freeze manifest in
`outputs/single_sku_evaluation/development/model_freeze_manifest.json`.
