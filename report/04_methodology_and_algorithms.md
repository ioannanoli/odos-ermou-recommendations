# 4. Methodology and Algorithms

## FP-Growth

A dependency-free FP-tree mines frequent itemsets. The production run uses
minimum support 0.002 and limits sets to three SKUs. The output reports itemset,
length, order count, and relative support.

## Configurable co-purchase graph

Each SKU is a node and each within-order pair is an undirected edge. The
original cosine association remains the default. Controlled alternatives add
raw pair-count pruning, Jaccard and lift edge weights, and directional
confidence for recommendation ranking. Optional half-life decay weights each
basket relative to the end of its training period, never a future development
or test date.

Direct cart recommendations aggregate neighbor evidence from every observed
cart SKU. Edge output retains raw and weighted pair counts, support,
confidence, Jaccard, lift, and the selected graph weight.

## Node2vec and Skip-Gram

Weighted second-order random walks convert graph neighborhoods into product
sequences. Return parameter `p` and exploration parameter `q` control walk
behavior. A local Skip-Gram model with negative sampling learns normalized SKU
vectors from center/context pairs. Exact cosine k-nearest-neighbor retrieval
uses the normalized mean vector of known cart products.

Nodes, neighbors, metadata memberships, and graph insertions are sorted before
seeded sampling. This prevents Python set or graph insertion order from changing
walks between separate processes with the same random seed.

## Adamic–Adar

Missing graph links are scored through shared neighbors:

```text
AA(i,j) = sum(1 / log(degree(z))) for each shared neighbor z
```

Low-degree shared neighbors contribute more specific evidence than universally
connected products.

## Metadata similarity

The explicit metadata signal uses weighted field-wise Jaccard overlap for
multi-valued category, brand, age, hero, and gender attributes. Unknown values
are ignored. This signal can introduce catalog-similar candidates without
turning metadata into a hard requirement; the existing metadata filter remains
available separately.

## Multi-signal recommendation engine

The engine min-max normalizes direct co-purchase, Product2Vec, Adamic–Adar, and
metadata candidate scores independently before applying tunable top-level
weights. Zero-weight sources are skipped. Candidate retrieval is cached during
weight search so identical source rankings are not recomputed.

## Product-name TF-IDF

The fourth production signal is a dependency-free sparse TF-IDF model over
`Product Name`. Names are Unicode-normalized and case-folded while letters and
product-code digits are retained. One- and two-word features capture phrases;
three- to five-character features handle spelling variants, Greek/English text,
and related model codes. Smoothed inverse document frequency and log-scaled term
frequency are L2-normalized. Candidate scores combine 60% word-vector cosine
and 40% character-vector cosine. An inverted feature index avoids comparing
every catalog pair.

The multi-signal engine normalizes TF-IDF candidate scores in the same way as
the behavioral and structured metadata signals.

## Heterogeneous metadata graph

The experimental graph contains typed product, category, brand, age, and hero
nodes. `Unknown` metadata is excluded. Product–metadata edge weights are divided
by the square root of metadata-node degree so broad hubs do not dominate walks.
Ordinary weighted walks traverse the mixed graph, but nearest-neighbor output
is restricted to product SKUs. This graph remains an explicit ablation rather
than silently replacing the original product graph.

## Graph visualizations

Full graphs contain thousands of nodes, so report plots use a 100-edge backbone
and a bounded two-hop ego graph. Node size/color represents graph degree or
query role, while edge width represents normalized co-purchase weight.

![Selected graph backbone](../outputs/improvement_experiments/plots/selected_graph_backbone.png)

![Selected graph ego network](../outputs/improvement_experiments/plots/selected_graph_ego_IT16951.png)

## Serving interface

`FinalRecommender` loads the frozen development-selected JSON and refits that
unchanged architecture on all historical orders available at deployment time.
The trained graph, embeddings, catalog, and hybrid are persisted as a trusted
local pickle so serving queries do not retrain the model. The Streamlit product
page uses one viewed SKU and presents one **Προϊόντα που μπορεί να σας αρέσουν**
shelf. Its current development-selected ranking is 40% direct co-purchase, 10%
Product2Vec, 10% structured metadata, and 40% product-name TF-IDF.
Recommendations can be restricted to
a current inventory SKU list, optionally filtered by metadata, and enriched
with catalog fields. The serving workflow does not rerun or modify the
historical test evaluation. The previous 40/30/30 test result remains a
historical comparison and is not relabeled as evidence for the TF-IDF model.
