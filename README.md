# Odos Ermou Toy Recommendation System

This project implements the first six phases of a local recommendation
pipeline: cleaning order data, mining frequent baskets, constructing a
co-purchase graph, generating Node2vec random walks, training product
embeddings with Skip-Gram negative sampling, and running chronological model
evaluation with random hyperparameter search, and a cart recommendation engine
combining embedding k-NN, Adamic–Adar link prediction, and metadata filters,
followed by quantitative evaluation and qualitative error analysis.

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

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Run the complete pipeline:

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
|   `-- phase6/
|       |-- summary_metrics.csv
|       |-- prediction_outcomes.csv
|       |-- segment_errors.csv
|       `-- qualitative_samples.csv
|-- src/
|   |-- data_loader.py
|   |-- evaluation.py
|   |-- error_analysis.py
|   |-- fp_growth.py
|   |-- copurchase_recommender.py
|   |-- product2vec_recommender.py
|   |-- model_config.py
|   `-- recommendation_engine.py
|-- tests/
|   `-- test_recommenders.py
|-- .gitignore
|-- main.py
|-- phase4_experiments.py
|-- phase6_analysis.py
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

- `build_copurchase_graph(orders)` creates one node per SKU and one undirected
  edge for each pair observed in the same order. Every edge stores its raw
  co-purchase `count` and cosine-normalized `weight`. Each node stores the
  number of orders containing that SKU.
- `CoPurchaseRecommender.__init__()` creates an empty graph and catalog slot.
- `CoPurchaseRecommender.fit(orders, product_catalog=None)` constructs the graph
  and returns the fitted model.
- `CoPurchaseRecommender.recommend(sku, top_n=10)` ranks direct graph neighbors
  by normalized weight and also reports pair count, support, directional
  confidence, and Jaccard similarity.

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
- `fit(orders, product_catalog=None)` constructs the graph, assigns matrix
  indices to SKUs, creates walks, trains embeddings, and returns the model.
- `recommend(sku, top_n=10)` calculates cosine similarity through the normalized
  embedding matrix and returns the closest connected products.
- `recommend_cart(cart_skus, top_n=10)` averages known cart embeddings,
  normalizes the cart vector, and performs exact cosine k-nearest-neighbor
  retrieval while excluding products already in the cart.

### `src/model_config.py`

- `SELECTED_PRODUCT2VEC_CONFIG` stores the configuration selected on the Phase
  4 development set so `main.py` and Phase 6 use exactly the same settings.

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
- The final `unittest.main()` block allows the test file to run directly.

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
- `requirements.txt` lists the four runtime packages. `openpyxl` reads the Excel
  file; `pandas` manages tables; `networkx` stores the graph; and `numpy` trains
  the embeddings.
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
