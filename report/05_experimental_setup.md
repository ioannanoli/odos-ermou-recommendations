# 5. Experimental Setup

## Chronological split

Complete orders are sorted by date and kept intact:

| Split | Orders | Lines | Unique SKUs | Multi-item orders | Period |
|---|---:|---:|---:|---:|---|
| Train | 6,808 | 10,310 | 4,797 | 1,810 | 2019-04-15 to 2024-07-13 |
| Development | 1,459 | 1,995 | 1,164 | 265 | 2024-07-14 to 2025-08-21 |
| Test | 1,460 | 1,873 | 1,035 | 216 | 2025-08-21 to 2026-06-19 |

This prevents a single order from appearing in multiple splits and ensures
model selection uses orders earlier than the test period.

## Controlled improvement search

Every candidate is fitted on train and ranked on 76 reproducibly generated
development basket-completion queries. Selection uses Hit Rate@10, followed by
MRR@10, Recall@10, and coverage. The search compares:

- four status policies derived from statuses actually present in the export;
- cosine, Jaccard, lift, and directional-confidence graph scoring;
- minimum pair counts of one, two, and three;
- no decay and 6, 12, 18, 24, and 36-month half-lives;
- 20 seeded Product2Vec configurations from the expanded search space;
- 85 valid top-level behavioral/metadata blends; and
- four hub-controlled heterogeneous relationship-weight configurations.

The original Product2Vec settings remained the best candidate: 48 dimensions,
walk length 8, two walks per node, window 3, two negative samples, two epochs,
learning rate 0.05, and `p = q = 1`.

The historical test-frozen architecture uses all statuses, the unpruned cosine graph, direct
confidence ranking, a six-month half-life, and a weighted heterogeneous graph.
Its final blend is 40% direct co-purchase, 30% Product2Vec, 30% metadata
similarity, and 0% Adamic–Adar. Metadata relationship weights are 0.05 each for
category, brand, age, and hero, versus 1.0 for co-purchase.

`python -m experiments.experiment_runner development` never evaluates test
queries. The separate
`finalize` stage reads the frozen JSON, refits on train+development, and writes
a sentinel result file. It refuses to evaluate the test split again once that
file exists.

## Final basket-completion protocol

The frozen model is refitted on train+development. For each eligible test
order, one known product is hidden reproducibly and the remaining products form
the cart. Each order contributes at most one query. Seventy-five test orders
have at least two products known to the fitted graph.

Metrics are Precision@10, Recall@10, Hit Rate@10, MRR@10, catalog coverage, and
bootstrap confidence intervals. Since item lines have no individual event
timestamps, the experiment evaluates basket completion—not next-item sequence
prediction.

## Relationship to product-page serving

Basket completion was the offline proxy used to select and compare the fitted
signals. Deployment places one combined recommendation shelf on a single
product page rather than on a cart page.

A separate product-page protocol now fits on the training period and evaluates
the already-frozen configuration on development orders. Each eligible order
contributes exactly one reproducibly selected directed pair: one viewed SKU and
one hidden basket partner. Both must be known to the training catalog. This
prevents large baskets from receiving more evaluation weight and ensures every
query has exactly one SKU, matching the interface. The same 76 queries compare
global popularity, category popularity, co-purchase, Product2Vec, metadata, and
product-name TF-IDF. It also compares the historical hybrid with 84 coarse
four-signal blends in 0.10 increments; all four signals must remain active.
Candidate recall@100 is recorded to separate retrieval
failures from top-ten ranking failures. The test split is not used by the
default command.

The selected product-page candidate is 40% co-purchase, 10% Product2Vec, 10%
structured metadata, and 40% TF-IDF. This configuration is selected and
reported on development data only. The previous test split is not reopened.

On 25 August 2026, the candidate was frozen. A freeze manifest records the
SHA-256 hashes of its configuration, 76 development queries, and complete
84-row weight search, plus the training cutoff of 19 June 2026 at 14:28:25.
Serving verifies the configuration hash before fitting. The development command
now rejects further tuning and rejects reopening the historical test split.

The preregistered next offline assessment used the unchanged model and every
eligible order strictly later than that cutoff. It rejects overlapping order
IDs, later rows in training history, altered configuration bytes, and reused
output directories. No hyperparameter or weight-selection option is available
in the future-period evaluator.

The assessment was executed once on 24 eligible queries dated from 19 June
through 25 August 2026. Frozen V1 and the already-defined expanded-candidate V2
were evaluated on exactly the same directed SKU pairs. Both used K=10,
candidate recall at 100, the same frozen 40/10/10/40 scoring formula, and 2,000
paired bootstrap samples. V2 changed retrieval breadth and source provenance
only; no weights or hyperparameters were selected using these future orders.
These 24 queries are now considered observed and closed to further tuning.

Its live business effect must still be measured on future data or in an A/B
test because offline co-occurrence does not measure customer interaction.

## Final parameter record

The submission separates four parameter types. The final ranking and metadata
weights are in `model_configs/final_product_page_weights.csv`; graph,
Node2vec, Skip-Gram, and TF-IDF settings are in
`model_configs/final_model_hyperparameters.csv`; the complete frozen JSON is in
`model_configs/final_product_page_model.json`; and parameters learned from the
9,727-order training snapshot are under `model_weights/`.

The learned export contains 6,257 graph nodes, 8,841 co-purchase edges, and
7,029 normalized 48-dimensional heterogeneous Node2vec vectors. The edge file
distinguishes the symmetric cosine weight used by random walks from directional
confidence used by direct co-purchase ranking. `export_manifest.json` records
the fitted pickle hash, frozen configuration hash, row counts, and file hashes.
These exports document the fitted model; they are not another tuning or
evaluation stage.

## Logistic-ranking backtest

The learned ranker uses only orders ending on 13 July 2024, before the frozen
76-query development period. Four whole-order windows contain 3,404 foundation
orders, 1,361 ranker-training orders, 1,021 validation orders, and 1,022 final
backtest orders. Candidate models are refitted at each boundary: snapshot 1
creates training features, snapshot 2 selects L2 from 0.01, 0.1, 1, and 10,
and snapshot 3 creates the untouched 70-query backtest. Thirty high-ranked
unpurchased candidates are sampled per training seed. The same candidate pools
are passed to the fixed and logistic rankers, and the backtest is evaluated
once using HR/Recall@10, MRR@10, candidate recall@100, full-pool recall,
coverage, popularity segments, paired wins/losses, and bootstrap intervals.
