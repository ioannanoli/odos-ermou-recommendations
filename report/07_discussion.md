# 7. Discussion and Limitations

## Interpretation

The controlled experiment shows that direct confidence-ranked co-purchase
evidence, recent-order emphasis, explicit metadata similarity, and a lightly
weighted heterogeneous graph are complementary. The final top-level
Adamic–Adar weight is zero, so it should not remain in the serving blend solely
because it was part of the earlier architecture.

For deployment, these fitted signals are combined in one product-page shelf.
This exposes both plausible alternatives and complements in the same ranked
list. The new development-selected candidate uses 40% co-purchase, 10%
Product2Vec, 10% structured metadata, and 40% product-name TF-IDF.

TF-IDF alone outperformed each earlier individual component on the dedicated
one-SKU development protocol. The four-signal blend raised HR@10 from 0.316 to
0.408, adding eight hits while losing one. Because its weight was
selected on these same 76 development queries, this is promising selection
evidence rather than an unbiased final estimate. On the later untouched period,
the frozen blend achieved HR@10 of 0.292 and MRR@10 of 0.225 across 24 eligible
queries. This confirms that the model can retrieve relevant products beyond
the development period, but the sample is too small for a precise estimate.

Expanded candidate generation raised full-pool recall on those future queries
from 0.792 to 1.000, while candidate recall@100 and HR@10 stayed unchanged at
0.458 and 0.292. Candidate breadth is therefore no longer the immediate
bottleneck on this sample; the added products need better source calibration or
reranking before V2 can improve what customers see.

Keeping all order statuses won on development data. In this export, cancelled
and pending baskets appear to retain useful shopping-intent information rather
than acting only as noise. Pair-count thresholds of two or three sharply hurt
performance because the graph is already sparse. The expanded Product2Vec
search did not beat the smaller original setup, showing that additional
training cost does not automatically produce better recommendations.

Metadata helped both coverage and rare-product retrieval. The heterogeneous
gain came from low metadata relationship weights with hub correction, rather
than allowing broad age or category nodes to dominate random walks.

The pre-freeze logistic experiment did not improve the final score. Although
the same candidate pool contained the hidden target for 80% of queries, the
learned ranker reduced HR@10 from 0.414 to 0.071. The small temporal learning
sample and exclusively hard-negative sampling produced unstable coefficients,
including negative Product2Vec and text-source effects. This demonstrates that
adding a learned layer is not automatically more sophisticated or accurate
than an interpretable fixed blend.

The submission exports the fitted graph and Node2vec matrix in addition to the
human-readable blend weights. This improves transparency: graph edge evidence
can be inspected directly, and any product or metadata-node embedding can be
reproduced from its 48 values. These parameters describe the frozen training
snapshot and must be regenerated, with a new manifest, after an approved
retraining run.

## Main weaknesses

- The final test contains only 75 eligible queries, so confidence intervals are
  wide and the popular segment has only three examples.
- Most eligible carts contain one observed product; more context remains the
  strongest predictor of success.
- Products absent from all training orders still require a content-only or
  business-rule fallback.
- Metadata is incomplete, repeated rows can disagree, and broad labels may not
  represent true substitutability.
- All-status usefulness may change as operational processes or WooCommerce
  status definitions change.
- Offline basket completion does not measure clicks, conversion, margin,
  availability, novelty, or customer satisfaction.
- Offline performance has not yet been confirmed with product-page interaction
  metrics.
- The same historical export has now supported several rounds of analysis;
  future confidence should come from genuinely new orders.
- The first genuinely future evaluation contains only 24 eligible queries and
  is now closed to tuning; its wide uncertainty prevents strong segment claims.
- The 70-order logistic backtest has now been observed and must not become a
  new tuning set through repeated sampling or feature changes.

## Recommended next work

1. Keep operating the persisted frozen model while the separately implemented
   Version 2 candidate generator retrieves a larger per-signal pool and records
   source provenance. The unlabeled candidate-health audit can monitor pool
   availability, invariants, source coverage, and latency now; validate ranking
   accuracy only on genuinely newer data.
   The completed audit covered all 6,257 catalog SKUs with no retrieval or
   invariant failures, a 1,327-product median pool, 38.8 ms median latency, and
   48.6 ms p95 latency. These figures measure engineering health rather than
   recommendation relevance.
2. Operate the persisted frozen model, retraining it on approved new order
   exports without changing its selected settings. Do not reopen tuning on the
   existing 76-query development set.
3. Feed the product page a current inventory list and add price, margin, and
   other business constraints.
4. Treat the completed 24-query future evaluation as closed. Accumulate a much
   larger later period or run a preregistered online experiment before making
   another model-selection decision.
5. Use metadata and TF-IDF as the fallback for products absent from behavioral
   training signals.
6. Collect impressions, clicks, add-to-cart actions, and purchases for the
   combined product-page shelf in an online A/B test.
7. Retrain on a schedule and monitor coverage, drift, status mix, latency, and
   popularity/cart-size segment performance. Regenerate the graph/embedding
   exports and hash manifest after each approved retraining snapshot.
8. Keep the logistic ranker experimental. If new interaction data provides a
   larger learning period, preregister mixed/random negative sampling or a
   pairwise objective before evaluating it on another untouched period.

## Conclusion

The historical frozen hybrid improved the one-time test point estimates for Hit Rate,
Recall, MRR, coverage, and the rare-product segment without reducing large-cart
Hit Rate. The TF-IDF product-page blend also produced seven top-ten hits on 24
genuinely future queries. Expanded retrieval found every future target but did
not improve top-ten ranking, so frozen V1 remains the serving choice. The small
sample, wide bootstrap intervals, and lack of online evaluation still require
continued monitoring and a larger later-period or online assessment.
