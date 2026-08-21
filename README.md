# Odos Ermou Toy Recommendation System

This project implements the first six phases of a local recommendation
pipeline: cleaning order data, mining frequent baskets, constructing a
co-purchase graph, generating Node2vec random walks, training product
embeddings with Skip-Gram negative sampling, and running chronological model
evaluation with random hyperparameter search, and a cart recommendation engine
combining embedding k-NN, Adamic–Adar link prediction, and metadata filters,
followed by quantitative evaluation and qualitative error analysis. A stricter
improvement pipeline now adds direct co-purchase blending, configurable graph
weights and pruning, time decay, status-policy comparison, explicit metadata
similarity, heterogeneous metadata walks, bootstrap intervals, and final plots.

No API key or external service is required. All processing runs locally.

The raw order workbook is intentionally excluded from Git because order exports
may contain customer or location information. After cloning the repository,
place the workbook at `data/Orders-Export-2026-June-07-2054.xlsx`, or pass a
different path to the relevant pipeline function/command.

## Setup

Python 3.10 or newer is recommended. From the project directory, install the
dependencies:

```powershell
python -m pip install -r requirements.txt
```

Alternatively, install the project and its nine console commands in editable
mode:

```powershell
python -m pip install -e .
```

This provides `odos-app`, `odos-recommend`, `odos-pipeline`, `odos-tune`, `odos-analyze`,
`odos-metadata-transfer`, `odos-visualize`, `odos-improve`, and
`odos-improvement-plots`.

## Train and use the final model

### Streamlit interface

After training the model at least once, launch the local browser interface:

```powershell
python -m streamlit run streamlit_app.py
```

If the project is installed in editable mode, the shorter command is:

```powershell
odos-app
```

The interface previews recommendations on a single product page. Search by SKU
or product name and it displays two separate sections: **Frequently bought
together**, ranked from direct historical co-purchases, and **Similar items**,
ranked from Product2Vec plus product metadata. It also supports inventory CSV or
Excel uploads, optional filters for similar items, score explanations, and UTF-8
CSV downloads. Products already shown under Frequently bought together are not
repeated under Similar items. It runs locally and does not need an API key.

The published HR/MRR figures evaluate the frozen combined hybrid with offline
basket completion. The two product-page sections are serving views over its
fitted components and need separate future-period or online evaluation.

### Command-line interface

Train the frozen architecture on all historical orders and save it locally:

```powershell
python serve_recommendations.py train
```

Recommend for a real cart after training:

```powershell
python serve_recommendations.py recommend IT16951 SKU-2 --top-n 10
```

Restrict results to SKUs listed in an inventory CSV or Excel workbook and save
the output:

```powershell
python serve_recommendations.py recommend IT16951 SKU-2 `
  --inventory inventory.csv --inventory-column SKU `
  --output outputs/serving/cart_recommendations.csv
```

Inspect the fitted model period and graph size:

```powershell
python serve_recommendations.py inspect
```

With an editable installation, replace `python serve_recommendations.py` with
`odos-recommend`. The saved `models/final_recommender.pkl` is ignored by Git;
only load this trusted local pickle or one produced by your own training job.

Python code can use the same service directly:

```python
from src.final_recommender import FinalRecommender

model = FinalRecommender.load("models/final_recommender.pkl")
recommendations = model.recommend(
    ["IT16951", "SKU-2"],
    top_n=10,
    available_skus={"SKU-3", "SKU-4", "SKU-5"},
)
```

Run the optional metadata-to-complement transfer experiment:

```powershell
python metadata_transfer_experiment.py
```

Generate graph visualizations, optionally centered on a particular SKU:

```powershell
python graph_visualizations.py
python graph_visualizations.py --center-sku IT16951
```

Run model selection on train/development data only:

```powershell
python experiment_runner.py development --trials 20
```

After reviewing and freezing `best_dev_configuration.json`, run the test stage
once:

```powershell
python experiment_runner.py finalize
```

The finalize command refuses to run if `final_test_results.csv` already exists.
Generate or refresh report plots without reevaluating any model:

```powershell
python visualization.py --center-sku IT16951
```

## Selected improvement result

The frozen final model uses all statuses, an unpruned cosine graph with direct
confidence ranking, six-month time decay, the original 48-dimensional
Product2Vec settings, and lightly weighted heterogeneous metadata edges. Its
top-level blend is 40% direct co-purchase, 30% Product2Vec, 30% metadata, and
0% Adamic–Adar.

| Test metric | Original P2V + AA | Selected final | Change |
|---|---:|---:|---:|
| Hit Rate / Recall@10 | 0.293 | 0.373 | +0.080 |
| MRR@10 | 0.112 | 0.154 | +0.041 |
| Coverage@10 | 0.068 | 0.076 | +0.008 |
| Rare-product Hit Rate@10 | 0.108 | 0.162 | +0.054 |

The test has only 75 eligible queries. The final Hit Rate@10 bootstrap interval
is 0.267–0.480, so the point gain is encouraging but should be validated on a
future order period.

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Run the older analytical/export pipeline:

```powershell
python main.py
```

Run Phase 4 experiments with the default four random trials and at most 250
evaluation queries per split:

```powershell
python phase4_experiments.py
```

For a quick smoke test, use one trial and 50 evaluation queries:

```powershell
python phase4_experiments.py --trials 1 --max-queries 50
```

Run Phase 6 leave-one-out evaluation and error analysis:

```powershell
python phase6_analysis.py
```

The default pipeline keeps every exported order status, including cancelled,
pending, refunded, and failed orders. Rows without an order ID or usable SKU
are removed.

## Project structure

```text
OdosErmouReccomendations/
|-- data/
|   `-- Orders-Export-2026-June-07-2054.xlsx
|-- outputs/
|   |-- frequent_itemsets.csv
|   |-- copurchase_recommendations.csv
|   |-- product2vec_recommendations.csv
|   |-- adamic_adar_recommendations.csv
|   |-- cart_recommendations.csv
|   |-- phase4/
|   |   |-- split_summary.csv
|   |   |-- baseline_metrics.csv
|   |   |-- hyperparameter_results.csv
|   |   `-- test_metrics.csv
|   |-- phase6/
|   |   |-- summary_metrics.csv
|   |   |-- prediction_outcomes.csv
|   |   |-- segment_errors.csv
|   |   `-- qualitative_samples.csv
|   |-- metadata_transfer/
|   |   |-- development_weight_search.csv
|   |   |-- summary_comparison.csv
|   |   |-- segment_comparison.csv
|   |   |-- prediction_comparison.csv
|   |   |-- rare_seed_comparison.csv
|   |   `-- improved_queries.csv
|   |-- improvement_experiments/
|   |   |-- baseline/
|   |   |-- status_policy/
|   |   |-- graph_weights/
|   |   |-- time_decay/
|   |   |-- product2vec_search/
|   |   |-- blend_search/
|   |   |-- heterogeneous_graph/
|   |   |-- final/
|   |   `-- plots/
|   `-- visualizations/
|       |-- graph_statistics.csv
|       |-- graph_backbone.png
|       |-- degree_distribution.png
|       |-- top_copurchase_edges.png
|       `-- sku_neighborhood_IT16951.png
|-- models/
|   `-- final_recommender.pkl
|-- src/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- error_analysis.py
|   |-- fp_growth.py
|   |-- copurchase_recommender.py
|   |-- product2vec_recommender.py
|   |-- model_config.py
|   |-- time_weighting.py
|   |-- metadata_recommender.py
|   |-- heterogeneous_graph.py
|   |-- final_recommender.py
|   |-- metadata_transfer_recommender.py
|   `-- recommendation_engine.py
|-- tests/
|   |-- test_final_recommender.py
|   |-- test_streamlit_app.py
|   |-- test_recommenders.py
|   `-- test_packaging.py
|-- report/
|   |-- README.md
|   |-- 01_introduction.md
|   |-- 02_data_collection.md
|   |-- 03_data_processing.md
|   |-- 04_methodology_and_algorithms.md
|   |-- 05_experimental_setup.md
|   |-- 06_results.md
|   `-- 07_discussion.md
|-- .gitignore
|-- main.py
|-- serve_recommendations.py
|-- streamlit_app.py
|-- phase4_experiments.py
|-- phase6_analysis.py
|-- metadata_transfer_experiment.py
|-- graph_visualizations.py
|-- experiment_runner.py
|-- visualization.py
|-- pyproject.toml
|-- README.md
`-- requirements.txt
```

## File and function reference

### `main.py`

This is the executable entry point. Its constants select the input workbook,
output folder, recommendation count, minimum FP-Growth support, and the
Product2Vec configuration selected during Phase 4.

- `_save_itemsets(orders, output_directory)` runs FP-Growth, converts each SKU
  tuple to a readable pipe-separated string, and writes
  `frequent_itemsets.csv`.
- `_save_recommendations(model, model_name, example_sku, product_catalog,
  output_directory)` asks one trained model for recommendations, joins product
  metadata, prints the results, and writes a model-specific CSV.
- `run_pipeline(data_path, output_directory)` orchestrates the complete flow:
  loading, cleaning, catalog creation, model training, basket mining, and CSV
  export. Its optional arguments allow a different workbook or output folder.
- The `if __name__ == "__main__"` block configures safe console output for Greek
  text and calls `run_pipeline()` only when the file is run directly.
- `main()` is the package console entry point used by `odos-pipeline`.

### `src/final_recommender.py`

This is the reusable production model wrapper.

- `load_frozen_configuration(path)` reads and validates the development-selected
  JSON without changing any settings.
- `FinalRecommender.from_config_path(path)` creates an unfitted service from the
  frozen configuration.
- `_training_orders(orders)` applies the selected status policy.
- `fit(orders)` trains the selected graph, heterogeneous Product2Vec, metadata
  similarity, and weighted hybrid on all supplied historical orders.
- `recommend(cart_skus, top_n, available_skus, metadata_filters, enrich)`
  returns enriched cart recommendations and optionally filters to currently
  sellable inventory.
- `recommend_frequently_bought_together(sku, ...)` creates a product-page
  complement section using direct historical co-purchases only.
- `recommend_similar(sku, ...)` creates a product-page alternatives section
  using an equal blend of Product2Vec and structured metadata.
- `save(path)` persists a trusted local model artifact; `load(path)` restores
  it without retraining.

### `serve_recommendations.py`

This is the production command behind `odos-recommend`.

- `load_available_skus(path, sku_column)` reads sellable SKUs from CSV or Excel.
- `train_model(data_path, config_path, model_path)` fits the frozen architecture
  on all historical orders and saves `models/final_recommender.pkl`.
- `recommend_from_model(...)` loads the saved model, applies optional inventory
  filtering, and optionally writes the recommendations to CSV.
- `build_parser()` defines the `train`, `recommend`, and `inspect` commands.
- `main()` executes the selected production command.

### `streamlit_app.py`

This is the interactive browser interface behind `odos-app`.

- `load_model(model_path, modified_ns)` caches the trusted trained model and
  refreshes it when its file changes.
- `read_inventory_file(file_name, content)` reads uploaded CSV or Excel stock
  lists while retaining textual SKUs and leading zeroes.
- `product_label(sku, catalog)` makes every picker option searchable by both
  SKU and product name.
- `metadata_values(catalog, field)` supplies clean values for optional filters.
- `prepare_results(recommendations)` creates the ranked, user-facing table.
- `main()` renders a searchable single-product page with separate Frequently
  bought together and Similar items sections, inventory controls, score charts,
  and CSV downloads.
- `launch()` starts Streamlit when the installed `odos-app` command is used.

### `phase4_experiments.py`

This is the Phase 4 entry point. It tunes Product2Vec without using the test set
for model selection.

- `SEARCH_SPACE` defines the discrete values sampled for embedding dimensions,
  walk settings, Skip-Gram settings, learning rate, and Node2vec `p` and `q`.
- `sample_configurations(n_trials, random_state)` draws unique configurations
  reproducibly instead of performing grid search.
- `_prefixed(metrics, prefix)` prepares train/dev metric names for one CSV row.
- `_diagnosis(train_hit_rate, dev_hit_rate)` labels a low training score as
  possible high bias and a large train/dev gap as possible high variance. This
  is a diagnostic hint, not the model-selection rule.
- `run_experiments(...)` creates chronological splits, evaluates the
  co-purchase baseline, runs random search on train/dev, selects by development
  Hit Rate then MRR and Recall, refits on train+dev, and evaluates test once.
- `parse_args()` defines `--data`, `--output`, `--trials`, `--max-queries`, and
  `--seed` command-line options.
- `main()` is the package console entry point used by `odos-tune`.

### `src/evaluation.py`

Contains the reusable Phase 4 splitting and metric logic.

- `METRIC_NAMES` lists the four ranking metrics averaged across queries.
- `chronological_order_split(orders, train_fraction, dev_fraction)` sorts whole
  orders by date and returns non-overlapping train, development, and test data
  frames. All lines from one order stay in the same split.
- `ranking_metrics(recommended, relevant, k)` calculates Precision@K, Recall@K,
  Hit Rate@K, and MRR@K for one seed product.
- `_evaluation_queries(orders, known_skus)` converts multi-product baskets into
  seed/relevant-product queries and excludes products unseen during fitting.
- `evaluate_recommender(model, orders, k, max_queries, random_state)` evaluates
  eligible queries, optionally takes a reproducible sample, caches repeated SKU
  recommendations, and reports ranking metrics, catalog coverage, and query
  counts.
- `split_summary(train, dev, test)` reports lines, orders, unique SKUs,
  multi-item baskets, and date boundaries for each split.

### `phase6_analysis.py`

This is the Phase 6 entry point.

- `run_analysis(...)` fits the selected Product2Vec model and Phase 5 engine on
  train+dev, hides one product from each eligible test basket, writes summary
  metrics, complete outcomes, segment analysis, and qualitative samples.
- `parse_args()` defines `--data`, `--output`, `--max-queries`, `--sample-size`,
  and `--seed` command-line options.
- `main()` is the package console entry point used by `odos-analyze`.

### `src/error_analysis.py`

Contains reusable Phase 6 loss, evaluation, segmentation, and inspection logic.

- `softmax_cross_entropy(logits, target_index)` computes stable categorical
  cross-entropy using the log-sum-exp transformation.
- `leave_one_out_queries(orders, known_skus, max_queries, random_state)` creates
  one reproducible hidden target and remaining cart from each eligible order.
- `_candidate_cross_entropy(recommendations, target_sku, epsilon)` calculates
  softmax loss among returned candidates and applies a finite penalty when the
  target is absent.
- `evaluate_cart_engine(...)` calculates Precision@K, Recall@K, Hit Rate@K,
  MRR@K, coverage, and candidate loss and returns every query outcome.
- `segment_errors(outcomes)` groups quality by target popularity and remaining
  cart size.
- `_metadata_tokens(value)` normalizes metadata into comparable tokens.
- `qualitative_samples(outcomes, product_catalog, sample_size, random_state)`
  samples predictions, joins target/prediction names, categories, and ages, and
  flags age or category mismatches for manual review.
- `bootstrap_ranking_intervals(outcomes, n_bootstrap, confidence,
  random_state)` reports query-bootstrap intervals for Hit Rate, Recall, and
  MRR.

The workbook identifies products within an order but does not provide a
separate timestamp for each order line. Phase 6 therefore evaluates basket
completion, not a falsely inferred next-item sequence. Its cross-entropy is a
candidate-set diagnostic and is not presented as full-catalog next-item loss.

### `src/data_loader.py`

Loads and normalizes the WooCommerce order export.

- `PRODUCT_COLUMNS` lists the SKU and metadata columns required to construct the
  product catalog.
- `TEXT_PRODUCT_COLUMNS` contains the metadata subset that must be cleaned.
- `MISSING_TEXT` defines text values treated as missing data.
- `clean_text(value)` decodes HTML entities, applies Unicode NFKC
  normalization, collapses repeated whitespace, and returns `pandas.NA` for
  missing-value markers.
- `load_orders(path, statuses=None)` reads the Excel workbook, validates its
  required columns, cleans text columns, removes rows without an order ID or
  SKU, and standardizes SKUs. With the default `statuses=None`, no order status
  is filtered. Pass an iterable such as `["wc-completed", "wc-cancelled"]` to
  use an explicit allow-list.
- `create_product_catalog(df)` creates one row per SKU. If repeated order lines
  contain conflicting metadata, the most common non-null value is selected;
  entirely missing attributes become `Unknown`.
- The nested `most_common(series)` helper inside `create_product_catalog`
  performs that metadata selection and imputation.

### `src/fp_growth.py`

Contains a dependency-free FP-Growth implementation.

- `_Node` is the internal FP-tree node. It stores an item, count, parent,
  children, and a link to the next node containing the same item.
- `_build_tree(weighted_transactions, min_count)` counts item frequency,
  removes infrequent items, orders the remaining items, and builds an FP-tree
  with its header links.
- `_mine(weighted_transactions, min_count, suffix, max_length)` recursively
  creates conditional pattern bases and yields frequent itemsets with counts.
- `frequent_itemsets(orders, min_support=0.01, max_length=None)` groups SKUs by
  order, converts relative or absolute support into a minimum count, mines the
  FP-tree, and returns `itemset`, `length`, `count`, and `support` columns.
  A float is interpreted as relative support; an integer is an order count.

### `src/copurchase_recommender.py`

Builds and queries the normalized product graph.

- `build_copurchase_graph(orders, weighting, min_pair_count,
  half_life_months, reference_date)` creates the configurable graph and stores
  raw/weighted counts plus cosine, Jaccard, lift, and selected weights.
- `CoPurchaseRecommender.__init__(...)` configures graph weighting, pruning,
  decay, and direct ranking.
- `CoPurchaseRecommender.fit(orders, product_catalog=None)` constructs the graph
  and returns the fitted model.
- `CoPurchaseRecommender.recommend(sku, top_n=10)` ranks direct graph neighbors
  by normalized weight and also reports pair count, support, directional
  confidence, and Jaccard similarity.
- `CoPurchaseRecommender.recommend_cart(...)` aggregates direct evidence across
  every cart SKU and caches repeated experiment queries.

### `src/product2vec_recommender.py`

Learns dense SKU vectors from the co-purchase graph.

- `Product2VecRecommender.__init__(...)` stores the embedding size, walk length,
  number of walks, context window, negative samples, epochs, learning rate,
  Node2vec `p`/`q` biases, and random seed. The small defaults keep the full
  workbook practical on a local computer.
- `_next_node(previous, current, rng)` samples the next weighted Node2vec step.
  `p` controls immediate returns and `q` controls outward exploration.
- `generate_walks()` generates reproducible second-order walks from every
  connected product node.
- `_positive_pairs(walks)` converts each walk into center/context training
  pairs using the configured context window.
- `_sigmoid(values)` computes a numerically stable logistic activation.
- `_train_skipgram(walks, rng)` trains input and output embedding matrices in
  batches with negative sampling, combines them, and L2-normalizes the result.
- `fit(orders=None, product_catalog=None, graph=None)` constructs the default
  graph or reuses a preconfigured graph before training embeddings.
- `recommend(sku, top_n=10)` calculates cosine similarity through the normalized
  embedding matrix and returns the closest connected products.
- `recommend_cart(cart_skus, top_n=10)` averages known cart embeddings,
  normalizes the cart vector, and performs exact cosine k-nearest-neighbor
  retrieval while excluding products already in the cart.

### `src/model_config.py`

- `SELECTED_PRODUCT2VEC_CONFIG` stores the configuration selected on the Phase
  4 development set so `main.py` and Phase 6 use exactly the same settings.

### `src/__init__.py`

- Marks `src` as an installable Python package and defines package
  `__version__ = "0.1.0"`.

### `src/recommendation_engine.py`

Contains Phase 5 retrieval, link prediction, metadata filtering, and blending.

- `AdamicAdarRecommender.__init__()` creates an empty product graph.
- `AdamicAdarRecommender.fit(orders=None, graph=None)` builds a graph from order
  lines or copies an existing fitted graph.
- `AdamicAdarRecommender.recommend(sku, top_n=10)` ranks products that are not
  directly connected to the SKU but share graph neighbors with it.
- `AdamicAdarRecommender.recommend_cart(cart_skus, top_n=10)` aggregates shared-
  neighbor evidence from every product in a cart.
- `_allowed_values(value)` converts one constraint or an iterable of constraints
  to a common list representation.
- `_metadata_matches(value, allowed)` performs case-insensitive whole-value and
  delimiter-separated token matching.
- `filter_by_metadata(recommendations, product_catalog, metadata_filters)` keeps
  candidates matching every requested field and rejects unknown columns.
- `RecommendationEngine.__init__(...)` stores the two models, catalog, and blend
  weights. Defaults assign 75% to k-NN and 25% to Adamic–Adar.
- `RecommendationEngine._normalize(frame, score_column)` min-max normalizes a
  source score before blending.
- `RecommendationEngine.recommend(cart_skus, top_n=10,
  metadata_filters=None)` retrieves candidates, filters them, blends both
  signals, and returns final cart recommendations.
- `HybridRecommendationEngine` generalizes this to independently normalized
  co-purchase, Product2Vec, Adamic–Adar, and metadata scores with named weights.

### `src/metadata_transfer_recommender.py`

Implements the optional rare-product complement-transfer experiment.

- `DEFAULT_FIELD_WEIGHTS` weights category, age, hero, brand, gender, and name
  similarity.
- `_metadata_tokens(value, product_name=False)` normalizes metadata and excludes
  missing markers.
- `MetadataTransferRecommender.__init__(...)` configures metadata neighbors,
  similarity threshold, purchase-reliability smoothing, and substitute penalty.
- `fit(orders=None, product_catalog=None, graph=None)` indexes metadata tokens
  and fits or reuses the co-purchase graph.
- `metadata_similarity(left_sku, right_sku)` calculates weighted field-wise
  Jaccard similarity.
- `_reliability(sku)` converts training order count into a smoothed reliability
  score.
- `similar_products(sku, top_n=None)` retrieves established metadata bridges
  through an inverted token index.
- `recommend(sku, top_n=10)` transfers bridge products' co-purchase complements
  and combines them with direct evidence.
- `recommend_cart(cart_skus, top_n=10)` aggregates transfer evidence across a
  cart.
- `MetadataEnhancedEngine` blends the production engine with normalized
  transferred-complement evidence.

### `metadata_transfer_experiment.py`

- `_fit_components(...)` fits common graph, embedding, and transfer components.
- `_evaluate_named(...)` evaluates one named engine consistently.
- `_rare_hit_rate(...)` extracts the hidden-rare-target segment metric.
- `evaluate_rare_seed_queries(...)` measures the intended use case where a rare
  SKU is the observed input and its basket partners are relevant complements.
- `run_experiment(...)` tunes transfer weight on development rare-seed metrics,
  locks the selected weight, and compares base/enhanced engines on test.
- `parse_args()` and `main()` provide the command-line interface.

### `graph_visualizations.py`

- `graph_statistics(graph)` reports graph size, density, components, isolates,
  largest component, and degree statistics.
- `strongest_edge_backbone(graph, max_edges)` selects the strongest observed
  edges so the global network remains readable.
- `neighborhood_subgraph(graph, center_sku, direct_limit, second_hop_limit)`
  selects a bounded weighted two-hop ego network.
- `_edge_widths(graph, minimum, maximum)` scales normalized edge weights into
  visible line widths.
- `plot_graph_backbone(...)`, `plot_sku_neighborhood(...)`,
  `plot_degree_distribution(...)`, and `plot_top_edges(...)` create four PNG
  charts using a non-interactive backend.
- `generate_visualizations(...)` builds the graph and writes all charts plus
  `graph_statistics.csv`.
- `parse_args()` and `main()` expose `--data`, `--output`, and `--center-sku`.

### `src/time_weighting.py`

- `order_time_weights(orders, half_life_months, reference_date)` returns one
  exponential-decay weight per order. Its default reference is the latest date
  inside the supplied training frame, which prevents future-date leakage.

### `src/metadata_recommender.py`

- `metadata_tokens(value)` parses multi-valued metadata and discards missing or
  `Unknown` markers.
- `MetadataSimilarityRecommender.fit(product_catalog)` indexes category, brand,
  age, hero, and gender tokens.
- `similarity(left_sku, right_sku)` computes weighted field-wise Jaccard
  similarity.
- `recommend_cart(cart_skus, top_n)` ranks SKU candidates by their strongest
  similarity to a product already in the cart.

### `src/heterogeneous_graph.py`

- `product_node(sku)` creates an unambiguous typed graph identifier.
- `build_heterogeneous_graph(product_graph, product_catalog,
  relationship_weights)` creates product, category, brand, age, and hero nodes,
  excludes unknown metadata, and down-weights hubs by inverse square-root
  degree.
- `HeterogeneousProduct2VecRecommender.fit(...)` trains ordinary weighted walks
  on the mixed graph.
- `recommend_cart(...)` and `recommend(...)` perform exact cosine retrieval but
  filter all output to real product SKUs.

### `experiment_runner.py`

- `inspect_status_policies(orders)` builds completed, successful-fulfilment,
  broader-intent, and all-status policies from observed values.
- `sample_product2vec_configurations(...)` returns 20 unique seeded candidates,
  including the current baseline.
- `blend_weight_grid(include_metadata)` creates valid coarse blends summing to
  one.
- `_fit_bundle(...)`, `_heterogeneous_bundle(...)`, `_engine(...)`, and
  `_evaluate(...)` fit and evaluate consistent component sets while recording
  complete experiment context.
- `run_development_search(...)` runs status, graph, decay, Product2Vec, blend,
  metadata, heterogeneous, and ablation comparisons using train/development
  only, then freezes `best_dev_configuration.json`.
- `run_final_evaluation(...)` refits the frozen winner on train+development,
  evaluates test once, writes outcomes, segments, baseline comparison, and
  bootstrap intervals, and refuses to overwrite an existing final result.
- `parse_args()` and `main()` expose the `development` and `finalize` stages.

### `visualization.py`

- `plot_model_comparison(...)` creates the development ablation grouped bars.
- `plot_final_baseline_comparison(...)` compares the original and frozen test
  results.
- `plot_segment_hit_rate(...)` plots popularity and cart-size segments.
- `plot_hyperparameter_search(...)` plots all Product2Vec development trials.
- `generate_experiment_visualizations(...)` writes the result plots plus a
  filtered selected-graph backbone and ego graph without model reevaluation.
- `parse_args()` and `main()` provide the plotting command.

### `tests/test_recommenders.py`

Defines a small deterministic three-order dataset and validates core behavior.

- `RecommenderTests.setUp()` creates the synthetic baskets before every test.
- `test_normalized_graph()` checks pair counts, normalization, and co-purchase
  ranking.
- `test_fp_growth()` checks frequent pair discovery and minimum support.
- `test_node2vec_is_reproducible()` fits two models with the same seed and
  verifies identical recommendations.
- `test_chronological_split_keeps_orders_intact()` checks chronological order,
  disjoint splits, and prevention of order leakage.
- `test_ranking_metrics()` verifies exact Precision, Recall, Hit Rate, and MRR
  calculations.
- `test_evaluation_counts_eligible_queries()` checks basket-to-query conversion
  and aggregate evaluation.
- `test_random_search_is_reproducible_and_unique()` checks seeded sampling.
- `test_adamic_adar_predicts_missing_link()` verifies a two-hop missing link and
  its exact score.
- `test_product2vec_recommends_for_cart()` verifies cart-level k-NN retrieval.
- `test_metadata_filter_requires_all_fields()` verifies multi-field filtering.
- `test_phase5_engine_blends_and_filters()` checks the final blended output.
- `test_softmax_cross_entropy()` verifies uniform and confident loss values.
- `test_leave_one_out_queries_are_reproducible()` verifies deterministic target
  hiding and prevents the target from remaining in the input cart.
- `test_qualitative_samples_flag_metadata_mismatch()` checks age/category error
  flags.
- `test_metadata_transfer_inherits_complement()` checks that an isolated rare
  LEGO product inherits a storage-box complement through a similar established
  LEGO product.
- `test_metadata_enhanced_engine_handles_isolated_product()` verifies the
  enhanced engine can serve that cold-start-style query.
- `test_graph_visualization_selection_and_statistics()` verifies graph summary,
  backbone, and bounded-neighborhood selection.
- The final `unittest.main()` block allows the test file to run directly.

### `tests/test_packaging.py`

- `PackagingTests.test_console_entry_points_resolve()` verifies all installed
  command targets are callable.
- `PackagingTests.test_pyproject_metadata_and_scripts()` checks the package name
  and five declared commands.
- `PackagingTests.test_analytical_report_has_all_sections()` ensures the full
  seven-chapter report is present.

### `report/`

The [analytical report](report/README.md) organizes the project as Introduction,
Data Collection, Data Processing, Methodology and Algorithms, Experimental
Setup, Results and Quantitative Analysis, and Discussion. Reported numbers are
copied from the versioned Phase 4 and Phase 6 output artifacts.

### Other files and directories

- `data/Orders-Export-2026-June-07-2054.xlsx` is the source WooCommerce order
  export. The code reads it but never modifies it.
- `outputs/frequent_itemsets.csv` contains frequent SKU sets, their sizes,
  counts, and support.
- `outputs/copurchase_recommendations.csv` contains graph-neighbor
  recommendations and product metadata.
- `outputs/product2vec_recommendations.csv` contains embedding-neighbor
  recommendations and product metadata.
- `outputs/adamic_adar_recommendations.csv` contains predicted missing links and
  common-neighbor evidence.
- `outputs/cart_recommendations.csv` contains blended Phase 5 recommendations,
  component scores, final scores, and product metadata.
- `outputs/phase4/split_summary.csv` documents the chronological split.
- `outputs/phase4/baseline_metrics.csv` contains train/dev co-purchase metrics.
- `outputs/phase4/hyperparameter_results.csv` contains every sampled Product2Vec
  configuration, train/dev metrics, generalization gap, and diagnostic.
- `outputs/phase4/test_metrics.csv` contains the final untouched-test results
  for co-purchase and the selected Product2Vec model.
- `outputs/phase6/summary_metrics.csv` contains overall basket-completion
  metrics and candidate loss.
- `outputs/phase6/prediction_outcomes.csv` contains every evaluated cart, hidden
  target, ranked predictions, rank, hit, loss, and analysis segments.
- `outputs/phase6/segment_errors.csv` compares popularity and cart-size groups.
- `outputs/phase6/qualitative_samples.csv` contains metadata-enriched random
  examples and mismatch flags for manual inspection.
- `outputs/metadata_transfer/` contains development weight selection, overall
  and segment comparisons, rare-seed results, all paired predictions, and
  queries improved by transfer.
- `outputs/visualizations/` contains the graph backbone, SKU neighborhood,
  degree distribution, top-edge chart, and numerical graph statistics.
- `outputs/improvement_experiments/` contains every development search table,
  the frozen configuration, ablation and segment results, the one-time final
  test outcomes, bootstrap intervals, baseline comparison, and final plots.
- `outputs/improvement_experiments/final/best_dev_configuration.json` is the
  complete frozen configuration selected without test feedback.
- `outputs/improvement_experiments/final/final_test_results.csv` is also the
  guard file that prevents accidental repeat test evaluation.
- `requirements.txt` lists the runtime packages. `openpyxl` reads Excel,
  `pandas` manages tables, `networkx` stores the graph, `numpy` trains the
  embeddings, and `matplotlib` creates static visualizations.
- `pyproject.toml` defines the installable package, Python requirement,
  dependencies, version, build backend, and console commands.
- `.gitignore` prevents Python caches, test caches, and local virtual
  environments from being tracked.
- `README.md` is this setup and code reference.

## Output columns

`frequent_itemsets.csv` contains:

- `itemset`: pipe-separated SKUs in the frequent set.
- `length`: number of SKUs in the set.
- `count`: number of orders containing the complete set.
- `support`: `count / total number of orders`.

`copurchase_recommendations.csv` contains the base/recommended SKUs, normalized
co-purchase score, pair count, support, confidence, Jaccard score, and joined
product metadata.

`product2vec_recommendations.csv` contains the base/recommended SKUs, embedding
cosine-similarity score, and joined product metadata.

`adamic_adar_recommendations.csv` contains link scores and shared-neighbor
counts. `cart_recommendations.csv` contains the cart, recommended SKU,
normalized k-NN and Adamic–Adar scores, final weighted score, and metadata.

## Metadata-filtered cart example

```python
recommendations = engine.recommend(
    ["SKU-1", "SKU-2"],
    top_n=10,
    metadata_filters={
        "Προϊόν Ηλικία": ["6+", "7+"],
        "Κατηγορίες προϊόντων": "LEGO",
    },
)
```

Values within one field are alternatives; separate fields must all match. Omit
`metadata_filters` to rank the full known catalog.

## Changing the defaults

- Change `TOP_N` in `main.py` to return more or fewer products.
- Change `MIN_ITEMSET_SUPPORT` to make FP-Growth more or less selective.
- Change `SELECTED_PRODUCT2VEC_CONFIG` only after a new Phase 4 experiment
  identifies a better development-set configuration.
- Pass `statuses` to `load_orders` if you later want to exclude particular
  order states.
- Increase `walk_length`, `walks_per_node`, or `epochs` when constructing
  `Product2VecRecommender` for more training at the cost of runtime.
- Adjust `p` and `q` to change whether random walks remain near the starting
  product or explore farther across the graph.
- Change the two `RecommendationEngine` weights to alter the balance between
  embedding similarity and missing-link evidence.
- Change Phase 4 runtime with `--trials` and `--max-queries`. Increasing either
  gives a broader comparison but takes longer.
- Change Phase 6 runtime with `--max-queries`; change the manual-review export
  size with `--sample-size`.
- Change improvement-search breadth with `experiment_runner.py development
  --trials N`; keep selection on development and do not alter the frozen model
  in response to `final_test_results.csv`.
- To run a genuinely new final evaluation, use a new future workbook and a new
  output directory. Do not delete the existing guard file merely to rerun the
  same test period.
- Retrain the serving artifact with `odos-recommend train` whenever a new order
  export is approved for deployment. This refits the frozen settings; it does
  not reopen model selection or reevaluate the existing test split.
