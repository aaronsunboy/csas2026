# Mixed Doubles Curling Power Play Analysis

## Repository Structure

scripts: python scripts
data/raw: Original provided CSV files
data/derived: derived datasets
outputs/posterior_mc: dynamic programming outputs
outputs/dp_point_estimate: point estimate outputs
outputs/diagrams: generated figures

## How to Run the Code

All commands should be run **from the repository root**.

```bash
python -u scripts/make_ends2.py
python -u scripts/plot_you_hammer_mean_heatmaps.py
python -u scripts/plot_you_hammer_mean_vs_sd.py
python -u scripts/plot_decision_tables.py
python -u scripts/plot_end_score_diff_distribution.py
```

All figures are written to outputs/diagrams