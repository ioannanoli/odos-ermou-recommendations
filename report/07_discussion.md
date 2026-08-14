# 7. Discussion and Limitations

## Interpretation

Graph embeddings broaden catalog coverage and slightly improve the probability
of retrieving a relevant product, while direct co-purchase relationships remain
valuable for placing successful recommendations early. The blended engine uses
both signals and can introduce products supported by shared-neighbor evidence
even when no direct co-purchase edge exists.

## Main weaknesses

- Rare products have insufficient graph evidence and low Hit Rate@10.
- Most eligible test carts contain only one observed context product.
- Cancelled, failed, and refunded orders may inject intent that differs from
  completed purchases.
- Cold-start SKUs unseen during fitting cannot be evaluated or embedded.
- Metadata is incomplete and repeated export rows can disagree.
- Candidate-set cross-entropy heavily penalizes absent targets but is not a
  full-catalog calibrated probability or a trained next-item loss.
- The test set contains only 75 eligible leave-one-out orders, and several
  segment estimates have small samples.

## Recommended next work

1. Compare completed-only training with the current all-status policy.
2. Add content embeddings for cold-start products.
3. Tune blend weights and metadata rules using development data only.
4. Add inventory, price, and business constraints before serving results.
5. Collect impression, click, add-to-cart, and purchase events for online A/B
   evaluation.
6. Retrain on a schedule and monitor coverage, drift, and segment performance.

## Conclusion

The project delivers a reproducible local baseline from raw order export to
recommendations, hyperparameter selection, untouched-test evaluation, and
manual error samples. Results support a hybrid graph/embedding approach, while
also showing that data sparsity and limited cart context—not simply model
choice—are the dominant constraints.
