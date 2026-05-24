# BurnoutSense AI Project Report

## Dataset Overview
- Rows analyzed: 2,000
- Target classes: High, Low, Medium
- Class balance: {'Low': 1019, 'Medium': 843, 'High': 138}

## Model Comparison
- Logistic Regression: accuracy=0.9225, precision=0.8700, recall=0.9436, f1=0.8996, cv_f1=0.9107±0.0097
- Random Forest: accuracy=0.9125, precision=0.8891, recall=0.9163, f1=0.9017, cv_f1=0.8932±0.0233
- XGBoost: accuracy=0.9250, precision=0.9242, recall=0.9255, f1=0.9247, cv_f1=0.9147±0.0068

## Best Model
- Selected model: XGBoost

## Behavioral Signals
- Burnout risk rises with longer work hours, heavier meeting load, and lower sleep efficiency.
- Productivity ratios and work-life balance scores improve separation between low and elevated burnout groups.
- Isolation intensity and after-hours work remain strong fatigue correlates.

## Deliverables
- Saved model artifact: models/burnout_predictor.pkl
- Exported metrics: reports/model_metrics.json
- Classification report: metrics/classification_report.txt
- Visual suite: visuals/*.png