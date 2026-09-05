# Odos Ermou Toy Recommendation System

This project implements a complete local recommendation pipeline: cleaning
order data, mining frequent baskets, constructing a
co-purchase graph, generating Node2vec random walks, training product
embeddings with Skip-Gram negative sampling, and running chronological model
evaluation with random hyperparameter search, and a cart recommendation engine
combining embedding k-NN, Adamic–Adar link prediction, and metadata filters,
followed by quantitative evaluation and qualitative error analysis. A stricter
improvement pipeline now adds direct co-purchase blending, configurable graph
weights and pruning, time decay, status-policy comparison, explicit metadata
similarity, heterogeneous metadata walks, bootstrap intervals, and final plots.

No API key or external service is required. All processing runs locally.

Raw order workbooks and CSV exports are intentionally excluded from Git because order exports
may contain customer or location information. After cloning the repository,
place the workbook at `data/Orders-Export-2026-June-07-2054.xlsx`, or pass a
different path to the relevant pipeline function/command.

## Setup

Python 3.10 or newer is recommended. From the project directory, install the
dependencies:

```powershell
python -m pip install -r requirements.txt
```

Alternatively, install the project and its fourteen console commands in editable
mode:

```powershell
python -m pip install -e .
```

To include the optional Word-report builder with an editable installation, use
`python -m pip install -e ".[report]"`.

This provides `odos-app`, `odos-recommend`, `odos-pipeline`, `odos-tune`, `odos-analyze`,
`odos-metadata-transfer`, `odos-visualize`, `odos-improve`, and
`odos-improvement-plots`, `odos-evaluate-product-page`, and
`odos-evaluate-future`, `odos-expanded-candidates`, `odos-candidate-health`, and
`odos-logistic-backtest`.

## Submission layout and final weights

Normal use is intentionally limited to `main.py`, `serve_recommendations.py`,
and `streamlit_app.py` in the repository root. Reusable implementation code is
under `src/`; all offline searches, evaluations, audits, and plotting programs
are under `experiments/`.

The selected models are summarized in `model_configs/model_registry.json`.
The current serving model is `model_configs/final_product_page_model.json`, and
its readable scalar weights are in
`model_configs/final_product_page_weights.csv`. The earlier historical
basket-completion model and its weights are preserved separately for accurate
reporting. These configuration and CSV files contain the selected scoring
weights. Graph/Node2vec/TF-IDF settings are in
`model_configs/final_model_hyperparameters.csv`; fitted graph edges and
Node2vec vectors are exported under `model_weights/`.

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

The interface previews one combined recommendation shelf on a single product
page. Search by SKU or product name and **Προϊόντα που μπορεί να σας αρέσουν**
ranks both similar and frequently co-purchased products using the product-page
blend: 40% co-purchase, 10% Product2Vec, 10% structured metadata, and 40%
TF-IDF product-name similarity. It also supports inventory
CSV or Excel uploads, optional metadata filters, score explanations, and UTF-8
CSV download. It runs locally and does not need an API key.

The 0.373 test HR reported below belongs to the earlier 40/30/30 frozen model.
The TF-IDF extension now also has a small strictly future evaluation: HR@10 is
0.292 across 24 eligible orders after the locked cutoff. Neither offline result
measures click-through or conversion on the live product-page shelf.

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

Export every learned graph edge and final Node2vec vector in readable form:

```powershell
python serve_recommendations.py export-weights
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
python -m experiments.metadata_transfer_experiment
```

Generate graph visualizations, optionally centered on a particular SKU:

```powershell
python -m experiments.graph_visualizations
python -m experiments.graph_visualizations --center-sku IT16951
```

Run model selection on train/development data only:

```powershell
python -m experiments.experiment_runner development --trials 20
```

After reviewing and freezing `best_dev_configuration.json`, run the test stage
once:

```powershell
python -m experiments.experiment_runner finalize
```

The finalize command refuses to run if `final_test_results.csv` already exists.
Generate or refresh report plots without reevaluating any model:

```powershell
python -m experiments.visualization --center-sku IT16951
```

## Historical frozen test result

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

## TF-IDF product-page candidate

The dedicated one-SKU development protocol gives the earlier 40/30/30 hybrid
HR@10 **0.316** after deterministic walk ordering. Product-name TF-IDF alone
also reaches **0.316**. A coarse search of 84 blends, with every signal kept
active, selected 40% co-purchase, 10% Product2Vec, 10% metadata, and 40% TF-IDF.
It reaches HR@10 **0.408**, MRR@10 **0.214**, candidate recall@100 **0.553**,
and coverage@10 **0.108** across 76 orders. It adds eight hits while losing one
earlier hit, for a net gain of seven. These are
development-selected figures, not a new test result.

This candidate is now frozen. The configuration hash is checked whenever the
default model is trained, and the existing development and historical test
periods are closed to further selection.

## Version 2 expanded candidate generation

Expanded retrieval is implemented as a separate experiment; it does not alter
the frozen Version 1 configuration or `models/final_recommender.pkl`. For a
normal top-10 request, Version 2 retrieves at least 500 candidates independently
from each active signal before merging and ranking them. It records whether
co-purchase, Product2Vec, metadata, or TF-IDF retrieved each candidate.

Train the separate experimental model:

```powershell
python -m experiments.expanded_candidate_generation train
```

Inspect the full retrieved pool and the final top ten for a SKU:

```powershell
python -m experiments.expanded_candidate_generation inspect IT16951 `
  --output outputs/v2_candidates/IT16951.csv
```

This diagnostic adds two distinct measurements. `candidate_recall` still asks
whether the hidden target is in the ranked top `candidate_k` positions, while
`candidate_pool_recall` asks whether any source retrieved it anywhere in the
full pre-ranking pool. Its completed future comparison is reported below; the
frozen 76 development queries must not be used for more tuning.

Audit candidate health without using any hidden targets:

```powershell
python -m experiments.candidate_health_report
```

For a quick 100-product smoke test:

```powershell
python -m experiments.candidate_health_report `
  --max-skus 100 `
  --output outputs/v2_candidate_health_smoke
```

The audit checks every selected catalog SKU for empty or short pools, source
contribution and overlap, duplicate or self recommendations, score bounds,
final-score arithmetic, deterministic output, and latency. It exports
`product_health.csv`, `popularity_health.csv`, `source_contributions.csv`,
`source_combinations.csv`, and `summary.json`. It deliberately reports
`accuracy_metrics_calculated: false`: unlabeled catalog checks cannot estimate
HR, Recall, or MRR.

The completed full-catalog audit covers all 6,257 SKUs. It found zero failed
queries, invariant violations, empty pools, or short top-10 results. The median
candidate pool contains 1,327 products; median latency is 38.8 ms and p95
latency is 48.6 ms. Across all product queries, 6,193 different SKUs appear in
at least one top ten, giving aggregate diagnostic coverage of 98.98%. This is a
retrieval-health result, not evidence that 98.98% of future purchases will be
predicted.

## Logistic-ranking historical backtest

The learned final-score experiment is implemented with dependency-free,
L2-regularized logistic regression. It uses three separately fitted temporal
snapshots ending before the frozen development period, so none of the 76 locked
queries are used. Run it with:

```powershell
python -m experiments.logistic_ranker_experiment
```

The untouched pre-freeze backtest contains 70 one-target product-page queries:

| Ranker | HR/Recall@10 | MRR@10 | Candidate recall@100 | Pool recall |
|---|---:|---:|---:|---:|
| Fixed 40/10/10/40 | 0.414 | 0.276 | 0.671 | 0.800 |
| Logistic regression | 0.071 | 0.033 | 0.343 | 0.800 |

The logistic ranker produced no unique hits, lost 24 fixed-blend hits, and
retained five shared hits. Equal pool recall proves that both rankers received
the same retrieved targets; logistic regression ordered them poorly. It learned
negative standardized effects for Product2Vec score and text-source presence,
consistent with a small, sampling-sensitive ranking dataset. This is a valid
negative result: the fixed blend remains the preferred ranker, and the 70
backtest orders are now considered observed rather than reused for more tuning.

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
python -m experiments.phase4_experiments
```

For a quick smoke test, use one trial and 50 evaluation queries:

```powershell
python -m experiments.phase4_experiments --trials 1 --max-queries 50
```

Run Phase 6 leave-one-out evaluation and error analysis:

```powershell
python -m experiments.phase6_analysis
```

Run the dedicated one-SKU product-page evaluation on development orders:

```powershell
python -m experiments.single_sku_evaluation
```

This command reproduces the locked development metrics without changing the
model. The selected configuration and 76-query set are now frozen. Attempts to
use `--tune-weights` or reopen `--split test` are rejected. The archived weight
search remains in `outputs/single_sku_evaluation/development/`.

Evaluate a genuinely newer WooCommerce Excel or CSV export with:

```powershell
python -m experiments.future_period_evaluation `
  --future data/Orders-Export-NEW.csv `
  --output outputs/future_evaluation/NEW-PERIOD `
  --include-expanded-v2
```

Every evaluated line must be strictly later than **2026-06-19 14:28:25**.
The command refuses overlapping order IDs, altered frozen weights, training
history beyond the cutoff, an empty eligible query set, or overwriting an
earlier evaluation. It performs no tuning. After editable installation, use
`odos-evaluate-future`.

The first strictly future comparison used all eligible orders after the cutoff
through 25 August 2026. The export contains 287 new orders, but 242 are
single-product orders and 21 additional multi-product orders contain fewer than
two SKUs known to frozen training, leaving 24 queries.

| Model | HR/Recall@10 | MRR@10 | Recall@100 | Full-pool recall | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Frozen Version 1 | 0.292 | **0.225** | 0.458 | 0.792 | **0.033** |
| Expanded candidates V2 | 0.292 | 0.224 | 0.458 | **1.000** | 0.033 |

Both models make the same seven top-ten hits. Expanded retrieval finds all 24
targets somewhere in its much larger pool, versus 19 for Version 1, but the
additional five remain below rank 100 and do not improve displayed results.
The fixed Version 1 ranking therefore remains selected. The HR@10 bootstrap
interval is [0.125, 0.500], so another larger future period or online test is
needed for a precise estimate; these 24 orders are now observed and closed to
further tuning.

The default pipeline keeps every exported order status, including cancelled,
pending, refunded, and failed orders. Rows without an order ID or usable SKU
are removed.

## Project structure

```text
OdosErmouReccomendations/
|-- data/
|   `-- Orders-Export-2026-June-07-2054.xlsx
|-- outputs/
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
|   |-- single_sku_evaluation/
|   |   `-- development/
|   |       |-- summary_metrics.csv
|   |       |-- prediction_outcomes.csv
|   |       |-- bootstrap_intervals.csv
|   |       |-- popularity_segments.csv
|   |       |-- queries.csv
|   |       |-- split_summary.csv
|   |       |-- four_signal_weight_search.csv
|   |       |-- best_product_page_configuration.json
|   |       |-- model_freeze_manifest.json
|   |       `-- run_configuration.json
|   |-- future_evaluation/
|   |   `-- 2026-06-19_after_cutoff_to_2026-08-25_v1_v2/
|   |       |-- summary_metrics.csv
|   |       |-- prediction_outcomes.csv
|   |       |-- bootstrap_intervals.csv
|   |       |-- popularity_segments.csv
|   |       |-- paired_outcomes.csv
|   |       |-- queries.csv
|   |       `-- run_configuration.json
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
|   |-- visualizations/
|   |   |-- graph_statistics.csv
|   |   |-- graph_backbone.png
|   |   |-- degree_distribution.png
|   |   |-- top_copurchase_edges.png
|   |   `-- sku_neighborhood_IT16951.png
|   |-- v2_candidate_health/
|   |   |-- product_health.csv
|   |   |-- popularity_health.csv
|   |   |-- source_contributions.csv
|   |   |-- source_combinations.csv
|   |   `-- summary.json
|   `-- logistic_ranker_backtest/
|       |-- backtest_summary.csv
|       |-- backtest_outcomes.csv
|       |-- regularization_search.csv
|       |-- logistic_coefficients.csv
|       |-- popularity_segments.csv
|       |-- bootstrap_intervals.csv
|       `-- run_configuration.json
|-- models/
|   |-- final_recommender.pkl
|   `-- experimental_v2_expanded_candidates.pkl
|-- model_configs/
|   |-- README.md
|   |-- final_product_page_model.json
|   |-- final_product_page_weights.csv
|   |-- final_model_hyperparameters.csv
|   |-- historical_basket_model.json
|   |-- historical_basket_weights.csv
|   `-- model_registry.json
|-- model_weights/
|   |-- README.md
|   |-- final_graph_nodes.csv
|   |-- final_graph_edges.csv
|   |-- final_node2vec_embeddings.csv
|   `-- export_manifest.json
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
|   |-- tfidf_recommender.py
|   |-- heterogeneous_graph.py
|   |-- final_recommender.py
|   |-- model_freeze.py
|   |-- model_export.py
|   |-- logistic_ranker.py
|   |-- metadata_transfer_recommender.py
|   `-- recommendation_engine.py
|-- tests/
|   |-- test_final_recommender.py
|   |-- test_single_sku_evaluation.py
|   |-- test_tfidf_recommender.py
|   |-- test_model_freeze_future.py
|   |-- test_streamlit_app.py
|   |-- test_recommenders.py
|   |-- test_logistic_ranker.py
|   `-- test_packaging.py
|-- report/
|   |-- README.md
|   |-- 01_introduction.md
|   |-- 02_data_collection.md
|   |-- 03_data_processing.md
|   |-- 04_methodology_and_algorithms.md
|   |-- 05_experimental_setup.md
|   |-- 06_results.md
|   |-- 07_discussion.md
|   |-- FULL_REPORT.md
|   `-- Odos_Ermou_Recommendation_Report.docx
|-- experiments/
|   |-- README.md
|   |-- __init__.py
|   |-- phase4_experiments.py
|   |-- phase6_analysis.py
|   |-- single_sku_evaluation.py
|   |-- future_period_evaluation.py
|   |-- expanded_candidate_generation.py
|   |-- candidate_health_report.py
|   |-- logistic_ranker_experiment.py
|   |-- metadata_transfer_experiment.py
|   |-- graph_visualizations.py
|   |-- experiment_runner.py
|   `-- visualization.py
|-- scripts/
|   `-- build_word_report.py
|-- .gitignore
|-- main.py
|-- serve_recommendations.py
|-- streamlit_app.py
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
  JSON without changing any settings and verifies the current serving
  configuration against its freeze hash.
- `FinalRecommender.from_config_path(path)` creates an unfitted service from the
  frozen configuration.
- `_training_orders(orders)` applies the selected status policy.
- `fit(orders)` trains the selected graph, heterogeneous Product2Vec, metadata
  similarity, product-name TF-IDF, and weighted hybrid on all supplied
  historical orders.
- `recommend(cart_skus, top_n, available_skus, metadata_filters, enrich)`
  returns enriched cart recommendations and optionally filters to currently
  sellable inventory.
- `recommend_frequently_bought_together(sku, ...)` and
  `recommend_similar(sku, ...)` expose component-only diagnostic views. The
  Streamlit page uses the validated combined `recommend(...)` ranking.
- `save(path)` persists a trusted local model artifact; `load(path)` restores
  it without retraining.

### `experiments/expanded_candidate_generation.py`

This command builds and inspects Version 2 without modifying the frozen model.

- `expanded_candidate_configuration(base_config_path)` copies Version 1's
  trained architecture and weights, removes its selection-result labels, and
  adds the independent expanded-retrieval settings.
- `train_experimental_model(data_path, model_path)` fits and saves the separate
  `models/experimental_v2_expanded_candidates.pkl` artifact.
- `candidate_pool_diagnostics(model, sku, ranking_top_n)` returns the complete
  merged candidate pool, per-source candidate counts, and final ranking.
- `build_parser()` defines the `train` and `inspect` commands.
- `main()` runs the command and optionally exports the complete pool to CSV.

### `experiments/candidate_health_report.py`

Audits Version 2 retrieval across the catalog without using evaluation labels.

- `_name_language(value)` identifies Greek-only, Latin-only, mixed-script, and
  missing/other product names for multilingual reliability checks.
- `_clear_candidate_caches(engine)` bounds memory during all-catalog runs.
- `_candidate_sources(pool)` parses the retrieval provenance attached to each
  candidate.
- `_pool_is_deterministic(left, right)` checks ordered SKUs and final scores
  across repeated identical requests.
- `audit_candidate_health(...)` calculates product, source, segment, invariant,
  coverage, and latency diagnostics without HR, Recall, or MRR.
- `run_candidate_health_report(...)` loads the experimental model and exports
  the CSV and JSON artifacts.
- `parse_args()` defines model, output, top-N, reproducible sample, and
  determinism options; `main()` runs the command behind `odos-candidate-health`.

### `src/logistic_ranker.py`

Implements the learned candidate score without an external ML dependency.

- `CandidateFeatureBuilder(model, training_orders, reliability_scale)` creates
  28 strictly past-derived behavioral, content, popularity, recency, metadata,
  source-provenance, missingness, and reliability-interaction features.
- `CandidateFeatureBuilder.transform(seed_sku, candidate_pool)` converts a
  complete retrieved pool into its ordered numerical feature matrix.
- `RegularizedLogisticRanker(...)` configures L2 strength and Newton-solver
  convergence settings.
- `fit(features, labels, sample_weight)` standardizes features, balances the
  positive and negative classes, and fits regularized logistic coefficients.
- `predict_proba(features)` returns learned candidate relevance probabilities;
  `coefficient_frame()` returns interpretable standardized coefficients.

### `experiments/logistic_ranker_experiment.py`

Runs the leakage-safe learned-ranking experiment entirely before frozen
development.

- `chronological_backtest_periods(...)` creates four non-overlapping whole-order
  periods: foundation, ranker training, validation, and untouched backtest.
- `build_ranker_training_examples(...)` uses purchased basket relationships as
  positives and top retrieved unpurchased candidates as hard negatives.
- `prepare_evaluation_queries(...)` creates one reproducible hidden target per
  eligible order and retains the complete shared candidate pool.
- `evaluate_prepared_queries(...)` compares fixed-score and learned-probability
  ordering using HR, MRR, candidate recall, pool recall, and coverage.
- `run_logistic_backtest(...)` fits three temporal candidate snapshots, selects
  L2 on validation, refits the ranker, evaluates the final historical period
  once, and writes metrics, coefficients, segments, intervals, and paired rows.
- `parse_args()` exposes negative count, L2 values, K, seed, and bootstrap
  settings; `main()` runs the `odos-logistic-backtest` command.

### `src/model_freeze.py`

Defines and enforces the product-page model freeze.

- `HISTORICAL_TEST_CONFIG_PATH`, `DEFAULT_CONFIG_PATH`, and
  `FREEZE_MANIFEST_PATH` identify the archived test model, current serving
  model, and freeze record.
- `sha256_file(path)` fingerprints binary bytes exactly and text with
  canonical LF newlines, so checks remain stable across operating systems.
- `load_freeze_manifest(path)` validates and loads the locked cutoff, hash, and
  tuning state.
- `verify_frozen_configuration(...)` rejects any content change to the
  selected product-page configuration.
- `verify_frozen_artifacts(...)` verifies the configuration, query set, and
  archived weight-search hashes together.
- `assert_development_tuning_open(...)` blocks further searches on the 76
  development queries.
- `assert_legacy_test_reuse_allowed(...)` prevents evaluating the TF-IDF
  candidate on the already-seen historical test period.

### `src/model_export.py`

Creates readable, reproducible exports from the trained final artifact.

- `graph_node_frame(model)` exports frequency and degree statistics.
- `graph_edge_frame(model)` exports raw/time-decayed counts, cosine, Jaccard,
  lift, selected edge weight, support, and both directional confidences.
- `node2vec_embedding_frame(model)` exports the retained normalized 48-value
  vector for every product and metadata walk node.
- `export_model_parameters(...)` writes the three CSVs and a SHA-256 manifest.

### `serve_recommendations.py`

This is the production command behind `odos-recommend`.

- `load_available_skus(path, sku_column)` reads sellable SKUs from CSV or Excel.
- `train_model(data_path, config_path, model_path)` fits the frozen architecture
  on all historical orders and saves `models/final_recommender.pkl`.
- `recommend_from_model(...)` loads the saved model, applies optional inventory
  filtering, and optionally writes the recommendations to CSV.
- `build_parser()` defines the `train`, `recommend`, `inspect`, and
  `export-weights` commands.
- The `export-weights` command writes all final graph edges, graph-node
  statistics, and normalized Node2vec vectors plus a hash manifest.
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
- `score_formula(blend_weights)` renders the active four-signal weights in the
  recommendation explanation instead of hard-coding an obsolete formula.
- `main()` renders a searchable single-product page with one combined
  **Προϊόντα που μπορεί να σας αρέσουν** section, inventory controls, a score
  chart, and CSV download.
- `launch()` starts Streamlit when the installed `odos-app` command is used.

### `experiments/phase4_experiments.py`

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
- `single_sku_queries(orders, known_skus, max_queries, random_state)` selects
  exactly one reproducible viewed-SKU/hidden-target pair per eligible order.
  Both products must exist in the training catalog.
- `evaluate_single_sku_engine(engine, queries, train_orders, catalog_skus, k,
  candidate_k)` evaluates a product-page ranker and returns overall metrics plus
  per-query ranks, hits, top-`candidate_k` recall, full candidate-pool recall,
  retrieval-source provenance, popularity, and predictions.
- `split_summary(train, dev, test)` reports lines, orders, unique SKUs,
  multi-item baskets, and date boundaries for each split.

### `experiments/single_sku_evaluation.py`

This is the dedicated product-page offline evaluation entry point.

- `PopularityBaseline(...)` constructs a global-popularity baseline or a
  category-first popularity baseline from training orders only.
- `PopularityBaseline.recommend(...)` excludes the viewed SKU and ranks the
  remaining training products by order frequency, optionally prioritizing
  products that share a category.
- `_component_engine(model, signal)` exposes one already-fitted signal at a
  time so component comparisons use identical learned artifacts.
- `_hybrid_engine(model, weights)` creates a four-signal view over the fitted
  components without retraining.
- `four_signal_weight_grid(step)` generates coarse blends that sum to one and
  keep co-purchase, Product2Vec, metadata, and TF-IDF active.
- `search_four_signal_weights(...)` evaluates those blends and ranks them by
  Hit Rate, MRR, candidate recall, and coverage.
- `_segment_metrics(outcomes)` reports Hit Rate, MRR, and candidate recall for
  rare, medium, and popular hidden targets.
- `run_evaluation(...)` performs the chronological split, fits only on the
  earlier period, creates shared one-SKU queries, evaluates baselines and
  components, optionally searches four-signal weights, and writes metrics,
  outcomes, bootstrap intervals, segments, queries, split details, selected
  configuration, and run configuration.
- `parse_args()` defines the data, output, split, frozen configuration, K,
  candidate-pool size, query cap, seed, and bootstrap options.
- `main()` is the console entry point used by
  `odos-evaluate-product-page`.

### `experiments/future_period_evaluation.py`

Provides the only offline assessment path for the frozen TF-IDF candidate.

- `strictly_future_orders(orders, cutoff)` retains only lines whose date is
  later than the frozen serving cutoff.
- `_ensure_empty_output(output_directory)` prevents a previous future run from
  being overwritten or silently repeated.
- `run_future_evaluation(...)` verifies configuration integrity, validates
  independent dates and order IDs, fits on frozen history, evaluates every
  eligible later order without tuning, and writes metrics, outcomes, intervals,
  segments, queries, and data fingerprints. With `include_expanded_v2=True`, it
  also evaluates the predefined expanded pool on the same queries and writes
  paired outcomes.
- `parse_args()` requires `--future` and exposes history, output, K,
  candidate-pool, seed, bootstrap, and the predefined V2 comparison flag; it
  deliberately exposes no tuning options.
- `main()` is the `odos-evaluate-future` console entry point.

### `experiments/phase6_analysis.py`

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
- `load_orders(path, statuses=None)` reads an Excel workbook or UTF-8 CSV, validates its
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

The production hybrid also accepts a `text` signal with the `text_score`
column, backed by the TF-IDF model below. Its
`_validate_candidate_generation(...)` method validates independent source
budgets, while `_source_limit(...)` separates retrieval depth from the number
of products displayed. `candidate_pool(...)` returns the complete merged,
scored pool and optional source provenance; `recommend(...)` applies the same
final-score ranking and truncates that pool to the requested result count.

### `src/tfidf_recommender.py`

Provides local product-name content similarity without an API or scikit-learn.

- `PRODUCT_NAME_COLUMN` identifies the catalog text field used by the model.
- `normalize_product_name(value)` applies Unicode NFKC normalization,
  case-folding, punctuation removal, and whitespace normalization while
  retaining multilingual letters and product-code digits.
- `_ngrams(values, lower, upper)` yields contiguous word or character n-grams.
- `TfidfNameRecommender.__init__(...)` configures 60% word and 40% character
  channels, word 1–2 grams, and character 3–5 grams by default.
- `_channel_counts(...)` counts one product name's features.
- `_fit_channel(...)` calculates smoothed IDF, log-scaled term frequency,
  L2-normalized sparse vectors, and an inverted feature index.
- `fit(product_catalog)` builds both TF-IDF channels.
- `similarity(left_sku, right_sku)` returns their weighted cosine similarity.
- `_recommend_one(sku)` retrieves candidates through the sparse indexes.
- `recommend_cart(cart_skus, top_n)` ranks by the maximum name similarity to
  any query SKU; `recommend(sku, top_n)` is its one-SKU wrapper.

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

### `experiments/metadata_transfer_experiment.py`

- `_fit_components(...)` fits common graph, embedding, and transfer components.
- `_evaluate_named(...)` evaluates one named engine consistently.
- `_rare_hit_rate(...)` extracts the hidden-rare-target segment metric.
- `evaluate_rare_seed_queries(...)` measures the intended use case where a rare
  SKU is the observed input and its basket partners are relevant complements.
- `run_experiment(...)` tunes transfer weight on development rare-seed metrics,
  locks the selected weight, and compares base/enhanced engines on test.
- `parse_args()` and `main()` provide the command-line interface.

### `experiments/graph_visualizations.py`

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

### `experiments/experiment_runner.py`

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

### `experiments/visualization.py`

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

### `tests/test_tfidf_recommender.py`

- Verifies Unicode-aware product-name normalization, nearest-name ranking,
  cosine ordering, query-SKU exclusion, and empty output for missing names.

### `tests/test_model_freeze_future.py`

- Verifies the repository configuration hash, tamper detection, closed tuning
  and historical-test guards, strict future-date filtering, and output
  overwrite protection.

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
- `outputs/single_sku_evaluation/development/` contains the dedicated
  product-page model comparison, every one-SKU query and outcome, bootstrap
  intervals, target-popularity segments, all 84 four-signal blends, the selected
  product-page configuration, split summary, and exact run settings.
- `outputs/single_sku_evaluation/development/model_freeze_manifest.json` locks
  the configuration, query set, search table, cutoffs, weights, and selected
  metrics with SHA-256 fingerprints.
- `outputs/future_evaluation/2026-06-19_after_cutoff_to_2026-08-25_v1_v2/`
  contains the completed untouched-period comparison of frozen V1 and the
  predefined expanded-candidate V2, including shared queries, per-query
  outcomes, bootstrap intervals, popularity segments, paired outcomes, and
  the exact run configuration. The private raw CSV remains outside Git.
- `outputs/metadata_transfer/` contains development weight selection, overall
  and segment comparisons, rare-seed results, all paired predictions, and
  queries improved by transfer.
- `outputs/visualizations/` contains the graph backbone, SKU neighborhood,
  degree distribution, top-edge chart, and numerical graph statistics.
- `outputs/improvement_experiments/` contains every development search table,
  the frozen configuration, ablation and segment results, the one-time final
  test outcomes, bootstrap intervals, baseline comparison, and final plots.
- `model_configs/historical_basket_model.json` is the preserved historical
  40/30/30 configuration selected without test feedback; its readable weights
  are in `historical_basket_weights.csv` in the same folder.
- `model_configs/final_product_page_model.json` is the current frozen serving
  configuration selected on the dedicated development protocol; its readable
  weights are in `final_product_page_weights.csv`.
- `model_configs/final_model_hyperparameters.csv` lists the selected graph,
  Node2vec, Skip-Gram, and TF-IDF settings without calling them learned weights.
- `model_weights/` contains all final graph nodes/edges, the retained
  48-dimensional Node2vec matrix, and a file/hash manifest.
- The configuration JSON files remaining under `outputs/` are experiment
  archives, while `model_configs/` is the submission-facing source of truth.
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

`model_weights/final_graph_edges.csv` contains all co-purchase edge evidence
and weights. `final_node2vec_embeddings.csv` contains one normalized
48-dimensional vector per heterogeneous graph node. `export_manifest.json`
records their exact row counts and SHA-256 hashes.

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
- Change improvement-search breadth only in a new preregistered experiment with
  `python -m experiments.experiment_runner development --trials N`; keep
  selection on a new development period and do not alter the frozen model in
  response to any already-observed result file.
- To run a genuinely new final evaluation, use a new future workbook and a new
  output directory. Do not delete the existing guard file merely to rerun the
  same test period.
- Retrain the serving artifact with `odos-recommend train` whenever a new order
  export is approved for deployment. This refits the frozen settings; it does
  not reopen model selection or reevaluate the existing test split.
