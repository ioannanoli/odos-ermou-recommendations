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
evidence rather than an unbiased final estimate.

Keeping all order statuses won on development data. In this export, cancelled
and pending baskets appear to retain useful shopping-intent information rather
than acting only as noise. Pair-count thresholds of two or three sharply hurt
performance because the graph is already sparse. The expanded Product2Vec
search did not beat the smaller original setup, showing that additional
training cost does not automatically produce better recommendations.

Metadata helped both coverage and rare-product retrieval. The heterogeneous
gain came from low metadata relationship weights with hub correction, rather
than allowing broad age or category nodes to dominate random walks.

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

## Recommended next work

1. Operate the persisted frozen model, retraining it on approved new order
   exports without changing its selected settings. Do not reopen tuning on the
   existing 76-query development set.
2. Feed the product page a current inventory list and add price, margin, and
   other business constraints.
3. Evaluate the locked model on orders strictly later than 19 June 2026 using
   `future_period_evaluation.py`; do not change weights after seeing results.
4. Use metadata and TF-IDF as the fallback for products absent from behavioral
   training signals.
5. Collect impressions, clicks, add-to-cart actions, and purchases for the
   combined product-page shelf in an online A/B test.
6. Retrain on a schedule and monitor coverage, drift, status mix, latency, and
   popularity/cart-size segment performance.

## Conclusion

The historical frozen hybrid improved the one-time test point estimates for Hit Rate,
Recall, MRR, coverage, and the rare-product segment without reducing large-cart
Hit Rate. The selected blend supports the product-page prototype, while the
small sample, wide bootstrap intervals, and lack of online evaluation still
require monitoring and future-period validation. The stronger TF-IDF candidate
has development evidence only and should be confirmed on genuinely new orders.
