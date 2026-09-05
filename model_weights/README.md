# Learned final-model parameters

These files are readable exports from the trusted
`models/final_recommender.pkl` artifact. They are different from the selected
scalar weights and hyperparameters under `model_configs/`.

- `final_graph_nodes.csv` contains product order frequency, time-decayed order
  frequency, degree, and weighted degree for 6,257 product nodes.
- `final_graph_edges.csv` contains all 8,841 fitted co-purchase edges. It
  includes raw and time-decayed pair counts, support, cosine, Jaccard, lift,
  the selected cosine edge weight, and directional confidence in both
  directions.
- `final_node2vec_embeddings.csv` contains the 48-dimensional normalized final
  vector for all 7,029 heterogeneous walk nodes: 6,257 products and 772
  category, brand, age, or hero nodes.
- `export_manifest.json` records row counts and SHA-256 hashes for the model
  artifact, frozen configuration, and exported files.

The implementation learns separate Skip-Gram input and output matrices during
training and retains their L2-normalized sum as the final Product2Vec/Node2vec
representation. That retained matrix is what the embedding CSV exports and
what cosine recommendation uses.

Regenerate these files after intentionally retraining the unchanged frozen
architecture:

```powershell
python serve_recommendations.py export-weights
```
