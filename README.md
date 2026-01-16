# Mixed Doubles Curling Power Play Analysis

## Repository Structure

The repo is organized as follows:
* scripts: python code
* data
    * raw: provided CSV files
    * derived: processed datasets
* outputs
    * posterior_mc: dynamic programming outputs
    * dp_point_estimate: point estimate outputs
    * diagrams: generated figures

## How to Run the Code

All commands should be run **from the repository root** in the following order:

```bash
python -u scripts/make_ends2.py
python -u scripts/plot_you_hammer_mean_heatmaps.py
python -u scripts/plot_you_hammer_mean_vs_sd.py
python -u scripts/plot_decision_tables.py
python -u scripts/plot_end_score_diff_distribution.py
```

All figures are written to outputs/diagrams