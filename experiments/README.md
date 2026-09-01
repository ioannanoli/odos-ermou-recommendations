# Experiments

This package contains offline research code. It is separate from the serving
application in the repository root and from reusable model implementations in
`src/`.

Run a module from the repository root with `python -m experiments.<module>` or
use the installed `odos-*` command listed in the main README.

## Groups

- Model selection: `phase4_experiments.py`, `experiment_runner.py`, and
  `single_sku_evaluation.py`.
- Historical evaluation and error analysis: `phase6_analysis.py` and
  `metadata_transfer_experiment.py`.
- Locked later-period evaluation: `future_period_evaluation.py`.
- Experimental retrieval/ranking: `expanded_candidate_generation.py`,
  `candidate_health_report.py`, and `logistic_ranker_experiment.py`.
- Figures: `graph_visualizations.py` and `visualization.py`.

The 76-query development set, historical test period, 70-query logistic
backtest, and 24-query future period have already been observed. Do not use
them for additional tuning. The production model remains the frozen
configuration in `model_configs/final_product_page_model.json`.
