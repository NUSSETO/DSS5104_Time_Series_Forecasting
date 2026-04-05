# DSS5104 — Deep Learning for Time-Series Forecasting

**GitHub:** https://github.com/NUSSETO/DSS5104_Time_Series_Forecasting

Benchmarks 9 forecasting models (6 DL + 3 baselines) across 3 datasets (M4, M5, Traffic) using walk-forward validation with 3 random seeds.

## Repository Layout

- `Code_v1/` — initial implementation; models grouped by type. Reference only.
- `Code_v2/` — **primary implementation**; one pipeline file per model, fully independent runs.
- `DSS5104_CA2_Report.pdf` — final report (5 pages).
- `Proposal.txt` — project proposal outlining dataset, model, and experiment choices.
- `Instruction.txt` — original assignment brief.

## Data Setup

Download the raw data and place it in a `Data/` folder **one level above** this repository (i.e. sibling to the `CA2/` folder):

```
Data/
├── M4/                  # M4 competition CSVs (Monthly-train.csv, Monthly-test.csv)
├── M5/                  # Walmart M5 sales data (sales_train_evaluation.csv, calendar.csv, sell_prices.csv)
└── Traffic.tsf          # Traffic TSF file (UCI / Monash Time Series Repository)
```

Data is excluded from version control via `.gitignore`.

## Quickstart (Code_v2 — primary)

```bash
pip install -r Code_v2/requirements.txt

# Run all 9 models sequentially (full experiment, ~4 hours on Apple M4)
cd Code_v2
python pipelines/run_all.py

# Smoke test (quick validation, ~5 min)
python pipelines/run_all.py --smoke-test

# Run a single model
python pipelines/run_patchtst.py

# Aggregate results and generate plots
python analysis/aggregate_results.py
python analysis/plot_results.py
```

Results are saved to `Code_v2/results/`.
