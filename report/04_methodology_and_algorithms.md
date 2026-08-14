# 4. Methodology and Algorithms

## FP-Growth

A dependency-free FP-tree mines frequent itemsets. The production run uses
minimum support 0.002 and limits sets to three SKUs. The output reports itemset,
length, order count, and relative support.

## Co-purchase graph

Each SKU is a node and each observed within-order pair is an undirected edge.
Direct recommendations rank neighboring nodes by normalized edge weight and
also report support, directional confidence, and Jaccard similarity.

## Node2vec and Skip-Gram

Weighted second-order random walks convert graph neighborhoods into product
sequences. Return parameter `p` and exploration parameter `q` control walk
behavior. A local Skip-Gram model with negative sampling learns normalized SKU
vectors from center/context pairs.

## Embedding k-nearest neighbors

For one SKU, exact cosine similarity ranks normalized product embeddings. For a
cart, known product vectors are averaged and normalized before the same exact
k-nearest-neighbor search. Products already in the cart are excluded.

## Adamic–Adar

Missing graph links are scored through shared neighbors:

```text
AA(i,j) = sum(1 / log(degree(z))) for each shared neighbor z
```

Low-degree shared neighbors contribute more because they provide more specific
evidence than universally connected products.

## Blended recommendation engine

The engine min-max normalizes candidate scores and combines 75% embedding k-NN
with 25% Adamic–Adar. Optional metadata filters require every specified field
to match, while multiple allowed values inside one field act as alternatives.
