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

The frozen architecture uses all statuses, the unpruned cosine graph, direct
confidence ranking, a six-month half-life, and a weighted heterogeneous graph.
Its final blend is 40% direct co-purchase, 30% Product2Vec, 30% metadata
similarity, and 0% Adamic–Adar. Metadata relationship weights are 0.05 each for
category, brand, age, and hero, versus 1.0 for co-purchase.

`experiment_runner.py development` never evaluates test queries. The separate
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
