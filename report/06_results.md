# 6. Results and Quantitative Analysis

## Original baselines

The original single-SKU Phase 4 test showed complementary strengths:

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MRR@10 | Coverage |
|---|---:|---:|---:|---:|---:|
| Co-purchase | 0.147 | 0.348 | 0.411 | 0.290 | 0.068 |
| Product2Vec | 0.151 | 0.365 | 0.426 | 0.276 | 0.115 |

The original Phase 6 cart engine blended 75% Product2Vec with 25%
Adamic–Adar. On 75 basket-completion test queries it achieved Precision@10
0.029, Recall/Hit Rate@10 0.293, MRR@10 0.112, and coverage 0.068.

## Controlled development results

Development selection used 76 queries. The consolidated ablation was:

| Development model | Hit Rate@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|
| Co-purchase | 0.303 | 0.149 | 0.062 |
| Product2Vec | 0.250 | 0.138 | 0.092 |
| Product2Vec + Adamic–Adar | 0.224 | 0.124 | 0.094 |
| Behavioral hybrid | 0.276 | 0.168 | 0.095 |
| + metadata similarity | 0.316 | 0.244 | 0.112 |
| + six-month decay | 0.355 | 0.249 | 0.109 |
| + heterogeneous Product2Vec | **0.395** | **0.255** | 0.110 |

The unpruned cosine graph won. Directional confidence improved development Hit
Rate from 0.276 to 0.303. A six-month half-life raised the controlled hybrid to
0.329 before metadata/heterogeneous tuning. All-status training outperformed
the three filtered policies. The original Product2Vec configuration won the
20-candidate search on Hit Rate and the specified tie-break order.

![Development ablation](../outputs/improvement_experiments/plots/model_comparison.png)

![Product2Vec search](../outputs/improvement_experiments/plots/product2vec_search.png)

## Dedicated one-SKU product-page evaluation

The historical frozen configuration was evaluated with one viewed SKU and one
hidden partner per development order. Product-name TF-IDF was then added and
84 four-signal blends were compared. The test split was not inspected.

| Development model | Hit Rate@10 | MRR@10 | Candidate recall@100 | Coverage@10 |
|---|---:|---:|---:|---:|
| Global popularity | 0.092 | 0.029 | 0.197 | 0.002 |
| Category popularity | 0.171 | 0.066 | 0.342 | 0.023 |
| Co-purchase | 0.237 | 0.129 | 0.237 | 0.049 |
| Product2Vec | 0.184 | 0.117 | 0.276 | 0.108 |
| Metadata | 0.224 | 0.089 | 0.513 | 0.102 |
| Product-name TF-IDF | 0.316 | 0.177 | 0.474 | 0.106 |
| Historical 40/30/30 hybrid | 0.316 | 0.165 | 0.539 | 0.106 |
| Tuned four-signal hybrid | **0.408** | **0.214** | **0.553** | **0.108** |

The selected 40/10/10/40 hybrid retrieves 42 of 76 hidden partners in its first
100 candidates and ranks 31 of 76 in the top ten. Its HR@10 bootstrap interval
is 0.303–0.513. It adds eight hits and loses one relative to the 24-hit
historical hybrid, a net gain of seven. The 11
retrieved targets below rank ten still identify re-ranking as an improvement
opportunity. These figures are development-selected and therefore require a
new-period confirmation.

The reported Product2Vec and four-signal results use deterministic sorted walk
traversal. A separate-process reproduction returned the same metrics.

These development results are now locked. Their configuration, queries, and
weight-search files are hash-recorded in `model_freeze_manifest.json`; further
tuning and reuse of the historical test period are blocked.

## Strictly future product-page evaluation

The unchanged model was then evaluated once on an independent export. The
strict cutoff and overlap checks left 287 new orders from 19 June through 25
August 2026. Twenty-four orders contained at least two products known to the
frozen catalog and therefore formed eligible one-SKU queries.

| Model | Precision@10 | Recall/HR@10 | MRR@10 | Candidate recall@100 | Full-pool recall | Coverage@10 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen V1 | 0.029 | 0.292 | **0.225** | 0.458 | 0.792 | **0.033** |
| Expanded-candidate V2 | 0.029 | 0.292 | 0.224 | 0.458 | **1.000** | 0.033 |

Both models placed 7 of 24 hidden targets in the top ten, and their successful
queries were identical. V2 made all 24 targets available somewhere in its full
pool, compared with 19 for V1, but its additional five targets remained below
rank 100. It therefore improved retrieval availability without improving the
displayed top-ten recommendations. V1 remains the selected production model;
V2 is not promoted or refrozen.

The 95% bootstrap interval for HR@10 is [0.125, 0.500] for both models. V1's
MRR@10 interval is [0.083, 0.392], versus [0.083, 0.390] for V2. The wide
intervals reflect the small 24-query sample, so this is useful directional
evidence rather than a precise performance estimate. These future queries are
now closed and must not become a tuning set.

## One-time frozen test result

After settings were frozen, the selected model was refitted on
train+development and evaluated once on the 75 test queries.

| Model | Precision@10 | Recall/HR@10 | MRR@10 | Coverage@10 |
|---|---:|---:|---:|---:|
| Original P2V + AA | 0.029 | 0.293 | 0.112 | 0.068 |
| Selected final model | **0.037** | **0.373** | **0.154** | **0.076** |
| Absolute change | +0.008 | +0.080 | +0.041 | +0.008 |

The final model retrieved 28 of 75 hidden products versus 22 for the original
baseline.

These figures evaluate the historical 40/30/30 hybrid, not the newer TF-IDF
product-page candidate. The 0.373 Hit Rate measures offline basket completion,
not click-through, add-to-cart rate, conversion, or revenue.

![Final test comparison](../outputs/improvement_experiments/plots/final_baseline_comparison.png)

## Final segment results

| Segment type | Segment | Queries | Hit Rate@10 | MRR@10 |
|---|---|---:|---:|---:|
| Popularity | Rare | 37 | 0.162 | 0.047 |
| Popularity | Medium | 35 | 0.629 | 0.280 |
| Popularity | Popular | 3 | 0.000 | 0.000 |
| Cart size | Small | 49 | 0.327 | 0.159 |
| Cart size | Medium | 20 | 0.400 | 0.098 |
| Cart size | Large | 6 | 0.667 | 0.292 |

Rare-product Hit Rate increased from 0.108 to 0.162. Medium-product Hit Rate
increased from 0.514 to 0.629. Large-cart Hit Rate stayed at 0.667, while
small-cart Hit Rate increased from 0.224 to 0.327.

![Hit rate by popularity](../outputs/improvement_experiments/plots/hit_rate_by_popularity.png)

![Hit rate by cart size](../outputs/improvement_experiments/plots/hit_rate_by_cart_size.png)

## Statistical stability

Bootstrap resampling over the 75 final queries produced:

| Metric | Estimate | 95% bootstrap interval |
|---|---:|---:|
| Hit Rate@10 | 0.373 | [0.267, 0.480] |
| Recall@10 | 0.373 | [0.267, 0.480] |
| MRR@10 | 0.154 | [0.095, 0.224] |

The intervals are wide because the evaluation has only 75 observations. The
point improvements are promising, but their future-period stability is not
guaranteed.

## Logistic-ranking historical backtest

L2=0.01 won the separate 75-query validation period. The final model was then
refitted with training and validation examples and evaluated once on 70 later
orders ending on 13 July 2024.

| Ranker | HR/Recall@10 | MRR@10 | Candidate recall@100 | Pool recall | Coverage@10 |
|---|---:|---:|---:|---:|---:|
| Fixed 40/10/10/40 | **0.414** | **0.276** | **0.671** | 0.800 | **0.112** |
| Logistic regression | 0.071 | 0.033 | 0.343 | 0.800 | 0.044 |

The fixed ranker hit 29 targets and the logistic ranker hit five. All five were
shared; logistic regression introduced no unique hits and lost 24 fixed-ranker
hits. Fixed and learned pool recall are both 0.800, confirming a reranking
failure rather than a candidate-generation difference. The learned model
assigned negative standardized coefficients to Product2Vec score and
text-source presence, and the median rank of retrieved targets deteriorated
from 7 to 163. Its HR@10 95% bootstrap interval is [0.014, 0.129], compared
with [0.300, 0.529] for the fixed ranker. The learned ranker is therefore
rejected; this negative result is preserved rather than retuned on the observed
70 backtest orders.

## Reproducible artifacts

Every development candidate records its seed, dates, statuses, graph settings,
Product2Vec configuration, blend and metadata weights, query count, and
metrics under `outputs/improvement_experiments/`. The frozen JSON, one-time test
outcomes, segments, baseline comparison, and bootstrap intervals are in its
`final/` directory.

The earlier metadata-complement transfer experiment remains versioned under
`outputs/metadata_transfer/`, but it was not used to choose this model because
its previous test results predated the stricter selection protocol.

Submission-facing parameter records are separate from result tables. The
selected scalar weights and hyperparameters are under `model_configs/`.
`model_weights/final_graph_edges.csv` preserves all 8,841 fitted edges, and
`model_weights/final_node2vec_embeddings.csv` preserves the 7,029 by 48 final
normalized embedding matrix. Their hashes and row counts are recorded in
`model_weights/export_manifest.json`.
