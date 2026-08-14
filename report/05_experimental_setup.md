# 5. Experimental Setup

## Chronological split

Complete orders are sorted by date and kept intact:

| Split | Orders | Lines | Unique SKUs | Multi-item orders | Period |
|---|---:|---:|---:|---:|---|
| Train | 6,808 | 10,310 | 4,797 | 1,810 | 2019-04-15 to 2024-07-13 |
| Development | 1,459 | 1,995 | 1,164 | 265 | 2024-07-14 to 2025-08-21 |
| Test | 1,460 | 1,873 | 1,035 | 216 | 2025-08-21 to 2026-06-19 |

This prevents a single order from appearing in multiple splits and ensures
model selection uses orders earlier than the untouched test period.

## Random hyperparameter search

Four reproducible configurations were sampled across embedding dimensions,
walk length, walks per node, context window, negative samples, epochs, learning
rate, `p`, and `q`. Selection used development Hit Rate@10, with MRR@10 and
Recall@10 as tie-breakers.

The selected configuration is 48 dimensions, walk length 8, two walks per
node, window 3, two negative samples, two epochs, learning rate 0.05, and
`p = q = 1`.

## Phase 6 basket-completion protocol

Models are refitted on train+development orders. For each eligible test order,
one known product is hidden reproducibly and the remaining products form the
cart. Each order contributes at most one query. Seventy-five test orders have
at least two products known to the fitted graph.

Metrics are Precision@10, Recall@10, Hit Rate@10, MRR@10, catalog coverage, and
candidate-set softmax cross-entropy. Since item lines have no individual event
timestamps, the experiment evaluates basket completion—not next-item sequence
prediction.
