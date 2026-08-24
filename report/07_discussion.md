# 7. Discussion and Limitations

## Interpretation

The controlled experiment shows that direct confidence-ranked co-purchase
evidence, recent-order emphasis, explicit metadata similarity, and a lightly
weighted heterogeneous graph are complementary. The final top-level
Adamic–Adar weight is zero, so it should not remain in the serving blend solely
because it was part of the earlier architecture.

For deployment, these fitted signals are combined in one product-page shelf.
This exposes both plausible alternatives and complements in the same ranked
list and uses the same 40/30/30 weighting selected on development data.

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
   exports without changing its selected settings.
2. Feed the product page a current inventory list and add price, margin, and
   other business constraints.
3. Evaluate on a new future period rather than tuning to the current test.
4. Add a metadata-only fallback for products absent from all training orders.
5. Collect impressions, clicks, add-to-cart actions, and purchases for the
   combined product-page shelf in an online A/B test.
6. Retrain on a schedule and monitor coverage, drift, status mix, latency, and
   popularity/cart-size segment performance.

## Conclusion

The frozen hybrid improved the one-time test point estimates for Hit Rate,
Recall, MRR, coverage, and the rare-product segment without reducing large-cart
Hit Rate. The selected blend supports the product-page prototype, while the
small sample, wide bootstrap intervals, and lack of online evaluation still
require monitoring and future-period validation.
