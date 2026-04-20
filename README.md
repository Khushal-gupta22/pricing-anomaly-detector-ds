# Dynamic Pricing Anomaly Detector

An end-to-end anomaly detection system that monitors dynamic pricing algorithms for failures — surge spikes, flash crashes, stuck multipliers, and logic violations — using an ensemble of ML models, statistical methods, and business rules.

## Overview

Dynamic pricing systems (Uber, Lyft, DoorDash, etc.) can silently fail in production: overcharging customers, underpricing during peak demand, or ignoring market signals entirely. This project builds the monitoring layer that catches these failures before they escalate.

**Detection approach:** Three-layer ensemble combining Isolation Forest (15 engineered features), statistical methods (Z-score, MAD, IQR, rolling stats), contextual baselines (time-of-day aware), 8 deterministic business rules, and Prophet time series forecasting.

**Key result:** Ensemble F1 = **0.60** on synthetic data (after threshold tuning via PR curve analysis) and **0.80** on real NYC taxi data.

## Quick Start

```bash
git clone https://github.com/yourusername/dynamic-pricing-anomaly-detector.git
cd dynamic-pricing-anomaly-detector
pip install -r requirements.txt

# Run the full pipeline
python run_pipeline.py

# Launch dashboard
streamlit run dashboards/app.py

# Run tests (43 tests)
pytest tests/ -v
```

### Other Commands

```bash
python run_pipeline.py --no-prophet          # Skip Prophet (faster)
python run_pipeline.py --sample 5000         # Quick test on subset
python run_threshold_analysis.py --sample 5000   # Reproduce hyperparameter tuning
python run_nyc_test.py                       # Validate on real NYC taxi data
```

## Results

### Synthetic Data (~12% anomaly rate)

| Method              | Precision | Recall | F1        |
| ------------------- | --------- | ------ | --------- |
| ML Ensemble (tuned) | 66.4%     | 55.6%  | **60.5%** |
| Business Rules      | 46.7%     | 36.3%  | 40.8%     |
| Combined            | 49.7%     | 57.0%  | 53.1%     |

### NYC Taxi Data (100K real trips)

| Method         | Precision | Recall | F1        |
| -------------- | --------- | ------ | --------- |
| ML Ensemble    | 83.3%     | 76.8%  | **79.9%** |
| Business Rules | 86.1%     | 69.1%  | 76.7%     |
| Combined       | 75.6%     | 80.7%  | 78.0%     |

### Tuning Impact

The original ensemble threshold (0.40) gave 96% precision but only 15% recall (F1=0.29). PR curve analysis revealed lowering to 0.20 doubles F1 to 0.60 — most of the gain came from threshold tuning, not model complexity. Full grid search over 96 configurations confirmed optimal params. See [`notebooks/03_threshold_analysis.ipynb`](notebooks/03_threshold_analysis.ipynb).

## Real-World Validation (NYC Taxi Data)

The primary dataset is synthetic (real pricing data is proprietary), so we validate on **NYC Taxi & Limousine Commission (TLC)** trip records — 100K+ real trips with genuine heavy-tailed fares, seasonal patterns, and data quality issues.

```bash
# Download NYC taxi data and run all detectors
python run_nyc_test.py
```

`src/real_data_adapter.py` transforms raw taxi data into the pipeline's schema — mapping trip fares to `final_price`, trip distance to `distance_miles`, computing effective surge from rate codes, and categorizing trips by distance (short/standard/long/premium) and zone (airport/downtown/suburbs). Natural anomalies (extreme fares, zero-distance trips, impossible speeds) are flagged automatically, with optional synthetic anomaly injection for controlled evaluation.

**Results on real data:**

| Method         | Precision | Recall | F1        |
| -------------- | --------- | ------ | --------- |
| ML Ensemble    | 83.3%     | 76.8%  | **79.9%** |
| Business Rules | 86.1%     | 69.1%  | 76.7%     |
| Combined       | 75.6%     | 80.7%  | 78.0%     |
| LOF            | —         | —      | 29%       |
| DBSCAN         | —         | —      | 18%       |

The ensemble performs significantly better on real data (F1=0.80) than synthetic (F1=0.60), likely because real-world anomalies (extreme fares, zero-distance trips) are more separable than synthetic gradual drift. LOF and DBSCAN struggled with the variable-density real distribution, hence confirming Isolation Forest as the right choice.

## Architecture

```
Data Generator / NYC Adapter
        │
    SQLite DB
        │
  ┌─────┼─────┐
  IF  Stats  Context    ← ML Ensemble (weighted vote)
  └─────┼─────┘
        │
  ┌─────┼─────┐
Rules  Prophet          ← Business rules + time series
  └─────┼─────┘
        │
  Alert Classifier
        │
  Streamlit Dashboard (5 pages)
```

## Project Structure

```
├── src/
│   ├── anomaly_detector.py     # IF, Statistical, Contextual, LOF, SVM, DBSCAN
│   ├── business_rules.py       # 8 deterministic rules
│   ├── pipeline.py             # 7-stage orchestrator
│   ├── data_generator.py       # Synthetic data (11 anomaly types, clustering)
│   ├── time_series.py          # Prophet forecasting
│   ├── alerting.py             # Severity classification
│   ├── database.py             # SQLite layer
│   ├── experiment_tracker.py   # JSONL experiment logging
│   └── real_data_adapter.py    # NYC taxi data adapter
├── tests/
│   └── test_pipeline.py        # 43 tests (unit + integration + edge cases)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_threshold_analysis.ipynb
├── dashboards/
│   └── app.py                  # Streamlit (5 pages)
├── config/
│   └── settings.yaml
├── run_pipeline.py
├── run_threshold_analysis.py
└── run_nyc_test.py
```

## Tech Stack & Design Decisions

| Component             | Technology                                                  | Why                                                                                                                                                                                                                                                    |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Anomaly Detection     | Isolation Forest, LOF, One-Class SVM, DBSCAN (scikit-learn) | Benchmarked 4 methods: IF won on precision/speed tradeoff. LOF is transductive (can't predict on new data), SVM is 6x slower for similar results, DBSCAN is eps-sensitive.                                                                             |
| Statistical Analysis  | Z-score, MAD, IQR (numpy/scipy)                             | Each handles different distributions: Z-score for Gaussian, MAD for heavy tails (which pricing data has kurtosis > 0), IQR for skewed data, rolling for temporal changes. No single method covers all.                                                 |
| Time Series           | Prophet                                                     | Captures daily + weekly seasonality with uncertainty intervals. Catches gradual drift that point-in-time detectors miss entirely.                                                                                                                      |
| Business Rules        | Custom Python                                               | Deterministic, interpretable, zero false negatives on invariants (negative prices should never happen regardless of what any ML model thinks).                                                                                                         |
| Storage               | SQLite                                                      | Zero-config, embedded, fast enough for 100K+ events. This is a monitoring system, not a data warehouse. In production you'd swap for Postgres or TimescaleDB.                                                                                          |
| Dashboard             | Streamlit + Plotly                                          | Interactive, fast to build, deployable to Streamlit Cloud. Plotly gives hover tooltips and zoom which matter for anomaly investigation.                                                                                                                |
| Experiment Tracking   | Custom JSONL logger                                         | Lightweight alternative to MLflow for a single-developer project. Logs params, metrics, metadata per run. Flat file = no server, no database, version-controllable.                                                                                    |
| Ensemble Strategy     | Weighted vote (IF 40%, Stats 35%, Context 25%)              | No single method catches everything. IF misses context-dependent anomalies. Z-scores fail with heavy tails. Rules can't detect novel failures. Ensemble reduces false negatives while rules provide a hard safety net.                                 |
| Vectorization         | numpy/pandas throughout                                     | Original `iterrows()` approach took minutes on 100K events. Fully vectorized operations run in seconds. Sub-minute latency matters for real-time monitoring.                                                                                           |
| Synthetic Data        | Custom generator with anomaly clustering                    | Real pricing data is proprietary. The generator models actual pricing behavior (sigmoid surge curves, supply-demand dynamics) and injects realistic failure modes in bursts of 5-20; how bugs actually manifest in production, not as isolated events. |
| Real-World Validation | NYC TLC Taxi Dataset                                        | Validates that feature engineering and thresholds work on real distributions with genuine heavy tails, seasonality, and data quality issues.                                                                                                           |

## License

MIT
