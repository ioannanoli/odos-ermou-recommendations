# 6. Results and Quantitative Analysis

## Phase 4 single-SKU test results

| Model | Precision@10 | Recall@10 | Hit Rate@10 | MRR@10 | Coverage |
|---|---:|---:|---:|---:|---:|
| Co-purchase | 0.147 | 0.348 | 0.411 | 0.290 | 0.068 |
| Product2Vec | 0.151 | 0.365 | 0.426 | 0.276 | 0.115 |

Product2Vec improves Hit Rate, Recall, and catalog coverage. Co-purchase has the
higher MRR, meaning its successful recommendations tend to occur slightly
earlier in the list.

## Phase 6 blended cart results

| Metric | Value |
|---|---:|
| Evaluated orders | 75 |
| Precision@10 | 0.029 |
| Recall@10 | 0.293 |
| Hit Rate@10 | 0.293 |
| MRR@10 | 0.112 |
| Catalog coverage@10 | 0.068 |
| Mean candidate cross-entropy | 18.594 |

Because each query has one relevant hidden target, Recall@10 and Hit Rate@10
are identical, while Precision@10 equals hits divided by ten recommendations.

## Error segments

| Segment type | Segment | Queries | Hit Rate@10 | MRR@10 |
|---|---|---:|---:|---:|
| Popularity | Rare | 37 | 0.108 | 0.035 |
| Popularity | Medium | 35 | 0.514 | 0.204 |
| Popularity | Popular | 3 | 0.000 | 0.000 |
| Cart size | Small | 49 | 0.224 | 0.107 |
| Cart size | Medium | 20 | 0.350 | 0.077 |
| Cart size | Large | 6 | 0.667 | 0.271 |

The most reliable pattern is additional cart context: large carts achieve a
66.7% hit rate, compared with 22.4% for one-item carts. Rare products are the
largest populated weakness. The popular segment contains only three queries,
so its zero score is not a stable general conclusion.

## Reproducible artifacts

Exact experiment rows are stored in `outputs/phase4/`. Phase 6 summary,
per-query outcomes, segments, and metadata-enriched qualitative samples are in
`outputs/phase6/`.
