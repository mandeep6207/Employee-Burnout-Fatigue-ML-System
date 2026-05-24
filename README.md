# BurnoutSense AI
## Employee Burnout and Workplace Fatigue Intelligence System

BurnoutSense AI is an end-to-end behavioral analytics and machine learning project that predicts employee burnout risk, analyzes workplace fatigue patterns, and generates workforce intelligence artifacts for reporting and review.

## Project Overview

The pipeline ingests workplace behavior data, cleans and normalizes the records, engineers fatigue and productivity signals, compares multiple classification models, and exports a trained artifact together with metrics, reports, and visual diagnostics.

## Dataset

The repository ships with a behavioral dataset in `data/wfh_burnout_dataset.csv`. The code also standardizes the dataset path to `data/burnout_dataset.csv` for reproducible execution.

Key fields include:

- `day_type`
- `work_hours`
- `screen_time_hours`
- `meetings_count`
- `breaks_taken`
- `after_hours_work`
- `app_switches`
- `sleep_hours`
- `task_completion`
- `isolation_index`
- `fatigue_score`
- `burnout_score`
- `burnout_risk`

## Behavioral Analytics Workflow

1. Data cleaning and type normalization
2. Missing value handling
3. Feature engineering
4. Exploratory analysis
5. Visual artifact generation
6. Model comparison across Logistic Regression, Random Forest, and XGBoost
7. Metrics export and report generation
8. Notebook-based analysis summary

## Engineered Features

The pipeline creates:

- `productivity_ratio`
- `workload_index`
- `fatigue_level`
- `sleep_efficiency`
- `work_life_balance_score`

The modeling stack excludes leakage-prone target proxies such as `burnout_score` from prediction features.

## Visualizations

The project exports the following figures into `visuals/`:

- `burnout_distribution.png`
- `workhour_vs_burnout.png`
- `fatigue_heatmap.png`
- `sleep_analysis.png`
- `productivity_analysis.png`
- `isolation_impact.png`
- `feature_importance.png`
- `confusion_matrix.png`

## Model Comparison

The training pipeline compares three classifiers with cross-validation and selects the best model by macro F1 score:

- Logistic Regression
- Random Forest
- XGBoost

Expected behavior is strong but not perfect performance. The pipeline is designed to avoid target leakage so reported metrics stay realistic.

## Results

Outputs are written to:

- `models/burnout_predictor.pkl`
- `reports/model_metrics.json`
- `reports/project_report.md`
- `metrics/classification_report.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python -m src.main
```

This generates the model, metrics, reports, notebook-ready outputs, and all visualizations.

## Future Scope

- Add time-based behavioral drift monitoring
- Expand the model with intervention recommendations
- Introduce SHAP-based explainability
- Ship a lightweight dashboard for HR and team leads

## Repository Structure

```text
burnoutsense-ai/
├── data/
├── notebooks/
├── models/
├── visuals/
├── reports/
├── metrics/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```
