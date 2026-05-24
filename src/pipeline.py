from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_COLUMNS,
    CV_FOLDS,
    DATA_DIR,
    ENGINEERED_COLUMNS,
    LEAKAGE_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    MODEL_FEATURE_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    STANDARD_DATA_PATH,
    TARGET_COLUMN,
    TARGET_LABEL_ORDER,
    TEST_SIZE,
    VISUALS_DIR,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight"})


@dataclass
class PipelineArtifacts:
    best_model_name: str
    best_pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_names: list[str]
    metrics: dict[str, Any]
    report_text: str


class BurnoutFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: Any = None) -> "BurnoutFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        data = X.copy()
        data = data.rename(
            columns={
                "work_hour": "work_hours",
                "screen_time": "screen_time_hours",
                "meetings": "meetings_count",
                "breaks_taken": "breaks_taken",
                "after_hour_app_switch": "app_switches",
                "sleep_hour": "sleep_hours",
            }
        )
        for column in LEAKAGE_COLUMNS:
            if column in data.columns:
                data = data.drop(columns=column)
        data = self._clean(data)
        data = self._engineer(data)
        return data[MODEL_FEATURE_COLUMNS]

    def _clean(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric_candidates = [
            "work_hours",
            "screen_time_hours",
            "meetings_count",
            "breaks_taken",
            "after_hours_work",
            "app_switches",
            "sleep_hours",
            "task_completion",
            "isolation_index",
        ]
        for column in numeric_candidates:
            if column in data.columns:
                data[column] = pd.to_numeric(data[column], errors="coerce")
        if "day_type" in data.columns:
            data["day_type"] = data["day_type"].astype(str).str.strip().str.title()
        return data

    def _engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = data.copy()
        work_hours = frame["work_hours"].fillna(frame["work_hours"].median())
        screen_time = frame["screen_time_hours"].fillna(frame["screen_time_hours"].median())
        meetings = frame["meetings_count"].fillna(frame["meetings_count"].median())
        breaks = frame["breaks_taken"].fillna(frame["breaks_taken"].median())
        after_hours = frame["after_hours_work"].fillna(frame["after_hours_work"].median())
        app_switches = frame["app_switches"].fillna(frame["app_switches"].median())
        sleep_hours = frame["sleep_hours"].fillna(frame["sleep_hours"].median())
        task_completion = frame["task_completion"].fillna(frame["task_completion"].median())
        isolation = frame["isolation_index"].fillna(frame["isolation_index"].median())

        productivity_ratio = task_completion / (work_hours + 1.0)
        workload_index = (
            work_hours
            + 0.35 * screen_time
            + 0.75 * meetings
            + 1.25 * after_hours
            + app_switches / 25.0
        )
        fatigue_level = np.clip(
            (work_hours + screen_time + meetings + after_hours * 2 + app_switches / 20.0 - sleep_hours - breaks * 0.5)
            / 4.5,
            0,
            10,
        )
        sleep_efficiency = np.clip(sleep_hours / (work_hours + screen_time + 1.0), 0, 2)
        work_life_balance_score = np.clip(
            (sleep_hours + breaks * 0.6 + task_completion / 25.0) / (work_hours + screen_time + after_hours * 2 + 1.0),
            0,
            2,
        )

        frame["productivity_ratio"] = productivity_ratio
        frame["workload_index"] = workload_index
        frame["fatigue_level"] = fatigue_level
        frame["sleep_efficiency"] = sleep_efficiency
        frame["work_life_balance_score"] = work_life_balance_score
        frame["day_type"] = frame["day_type"].fillna("Weekday").astype(str).str.strip().str.title()
        for column in [
            "work_hours",
            "screen_time_hours",
            "meetings_count",
            "breaks_taken",
            "after_hours_work",
            "app_switches",
            "sleep_hours",
            "task_completion",
            "isolation_index",
        ]:
            frame[column] = frame[column].fillna(frame[column].median())
            frame[column] = frame[column].clip(lower=0)
        frame["day_type"] = frame["day_type"].where(frame["day_type"].isin(["Weekday", "Weekend"]), "Weekday")
        return frame


def resolve_data_path() -> Path:
    if STANDARD_DATA_PATH.exists():
        return STANDARD_DATA_PATH
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    raise FileNotFoundError("No burnout dataset found in data/.")


def load_raw_data() -> pd.DataFrame:
    path = resolve_data_path()
    return pd.read_csv(path)


def clean_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    cleaner = BurnoutFeatureEngineer()
    transformed = cleaner.transform(raw_data)
    result = raw_data.copy()
    rename_map = {
        "work_hour": "work_hours",
        "screen_time": "screen_time_hours",
        "meetings": "meetings_count",
        "after_hour_app_switch": "app_switches",
        "sleep_hour": "sleep_hours",
    }
    result = result.rename(columns=rename_map)
    for column in [TARGET_COLUMN, *LEAKAGE_COLUMNS]:
        if column in result.columns:
            result[column] = result[column]
    for column in transformed.columns:
        result[column] = transformed[column].values
    return result


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [column for column in MODEL_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def build_models() -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.9,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=280,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.35,
            reg_lambda=1.1,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=3,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
    return models


def build_pipeline(model: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineer", BurnoutFeatureEngineer()),
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )


def prepare_data() -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    raw = load_raw_data()
    cleaned = clean_dataset(raw)
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(cleaned[TARGET_COLUMN].astype(str))
    features = cleaned.drop(columns=[TARGET_COLUMN])
    return features, pd.Series(target, index=features.index), label_encoder


def compute_feature_names(pipeline: Pipeline) -> list[str]:
    transformed = pipeline.named_steps["preprocess"].get_feature_names_out()
    return [name.replace("numeric__", "").replace("categorical__", "") for name in transformed]


def evaluate_model(name: str, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, label_encoder: LabelEncoder) -> dict[str, Any]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro", zero_division=0)
    recall = recall_score(y_test, predictions, average="macro", zero_division=0)
    f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    return {
        "name": name,
        "pipeline": pipeline,
        "predictions": predictions,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "classification_report": report,
    }


def plot_burnout_distribution(data: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    order = [label for label in TARGET_LABEL_ORDER if label in data[TARGET_COLUMN].value_counts().index]
    sns.countplot(data=data, x=TARGET_COLUMN, order=order, palette="viridis")
    plt.title("Burnout Risk Distribution")
    plt.xlabel("Burnout Risk")
    plt.ylabel("Employee Count")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "burnout_distribution.png")
    plt.close()


def plot_workhour_vs_burnout(data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET_COLUMN, y="work_hours", palette="coolwarm")
    plt.title("Work Hours vs Burnout Risk")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "workhour_vs_burnout.png")
    plt.close()


def plot_fatigue_heatmap(data: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    numeric_columns = [
        "work_hours",
        "screen_time_hours",
        "meetings_count",
        "breaks_taken",
        "after_hours_work",
        "app_switches",
        "sleep_hours",
        "task_completion",
        "isolation_index",
        "fatigue_level",
        "workload_index",
        "productivity_ratio",
        "sleep_efficiency",
        "work_life_balance_score",
    ]
    corr = data[numeric_columns].corr(numeric_only=True)
    sns.heatmap(corr, cmap="RdYlBu_r", center=0, square=True)
    plt.title("Fatigue Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "fatigue_heatmap.png")
    plt.close()


def plot_sleep_analysis(data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="sleep_hours", y="burnout_score", hue=TARGET_COLUMN, alpha=0.75, palette="viridis")
    plt.title("Sleep Duration vs Burnout Score")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "sleep_analysis.png")
    plt.close()


def plot_productivity_analysis(data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x="work_hours", y="task_completion", hue=TARGET_COLUMN, alpha=0.75, palette="plasma")
    plt.title("Productivity Pattern Analysis")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "productivity_analysis.png")
    plt.close()


def plot_isolation_impact(data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=TARGET_COLUMN, y="isolation_index", palette="magma")
    plt.title("Isolation Impact on Burnout Risk")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "isolation_impact.png")
    plt.close()


def plot_model_feature_importance(pipeline: Pipeline, feature_names: list[str]) -> None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        scores = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        scores = pd.Series(coef.mean(axis=0), index=feature_names).sort_values(ascending=False).head(15)
    else:
        scores = pd.Series(dtype=float)
    if scores.empty:
        return
    plt.figure(figsize=(9, 6))
    sns.barplot(x=scores.values, y=scores.index, palette="crest")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "feature_importance.png")
    plt.close()


def plot_confusion(y_true: pd.Series, y_pred: np.ndarray, label_encoder: LabelEncoder) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "confusion_matrix.png")
    plt.close()


def build_metrics_payload(results: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {result["name"]: {key: result[key] for key in ["accuracy", "precision", "recall", "f1", "cv_f1_mean", "cv_f1_std"]} for result in results}
    best = max(results, key=lambda item: (item["f1"], item["accuracy"]))
    payload["best_model"] = best["name"]
    payload["best_model_f1"] = best["f1"]
    payload["best_model_accuracy"] = best["accuracy"]
    return payload


def build_report(data: pd.DataFrame, metrics_payload: dict[str, Any], label_encoder: LabelEncoder, best_name: str) -> str:
    class_counts = data[TARGET_COLUMN].value_counts().to_dict()
    report = [
        "# BurnoutSense AI Project Report",
        "",
        "## Dataset Overview",
        f"- Rows analyzed: {len(data):,}",
        f"- Target classes: {', '.join(label_encoder.classes_)}",
        f"- Class balance: {class_counts}",
        "",
        "## Model Comparison",
    ]
    for model_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if model_name in metrics_payload:
            values = metrics_payload[model_name]
            report.append(
                f"- {model_name}: accuracy={values['accuracy']:.4f}, precision={values['precision']:.4f}, recall={values['recall']:.4f}, f1={values['f1']:.4f}, cv_f1={values['cv_f1_mean']:.4f}±{values['cv_f1_std']:.4f}"
            )
    report.extend([
        "",
        f"## Best Model",
        f"- Selected model: {best_name}",
        "",
        "## Behavioral Signals",
        "- Burnout risk rises with longer work hours, heavier meeting load, and lower sleep efficiency.",
        "- Productivity ratios and work-life balance scores improve separation between low and elevated burnout groups.",
        "- Isolation intensity and after-hours work remain strong fatigue correlates.",
        "",
        "## Deliverables",
        "- Saved model artifact: models/burnout_predictor.pkl",
        "- Exported metrics: reports/model_metrics.json",
        "- Classification report: metrics/classification_report.txt",
        "- Visual suite: visuals/*.png",
    ])
    return "\n".join(report)


def execute_pipeline() -> PipelineArtifacts:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_raw_data()
    processed_data = clean_dataset(raw_data)
    X = processed_data.drop(columns=[TARGET_COLUMN])
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(processed_data[TARGET_COLUMN].astype(str)), index=processed_data.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model_results: list[dict[str, Any]] = []
    for model_name, model in build_models().items():
        pipeline = build_pipeline(model)
        result = evaluate_model(model_name, pipeline, X_train, y_train, X_test, y_test, label_encoder)
        model_results.append(result)

    best_result = max(model_results, key=lambda item: (item["f1"], item["accuracy"]))
    best_pipeline: Pipeline = best_result["pipeline"]
    feature_names = compute_feature_names(best_pipeline)

    plot_burnout_distribution(processed_data)
    plot_workhour_vs_burnout(processed_data)
    plot_fatigue_heatmap(processed_data)
    plot_sleep_analysis(processed_data)
    plot_productivity_analysis(processed_data)
    plot_isolation_impact(processed_data)
    plot_model_feature_importance(best_pipeline, feature_names)
    plot_confusion(y_test, best_result["predictions"], label_encoder)

    artifact_bundle = {
        "model_name": best_result["name"],
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "metrics": build_metrics_payload(model_results),
    }
    joblib.dump(artifact_bundle, MODELS_DIR / "burnout_predictor.pkl")

    metrics_payload = build_metrics_payload(model_results)
    metrics_payload["classification_report"] = best_result["classification_report"]
    with (REPORTS_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)
    with (METRICS_DIR / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(best_result["classification_report"])

    report_text = build_report(processed_data, metrics_payload, label_encoder, best_result["name"])
    (REPORTS_DIR / "project_report.md").write_text(report_text, encoding="utf-8")

    return PipelineArtifacts(
        best_model_name=best_result["name"],
        best_pipeline=best_pipeline,
        label_encoder=label_encoder,
        feature_names=feature_names,
        metrics=metrics_payload,
        report_text=report_text,
    )


def pretty_metrics_summary(metrics: dict[str, Any]) -> str:
    best = metrics.get("best_model", "unknown")
    acc = metrics.get("best_model_accuracy", 0.0)
    f1 = metrics.get("best_model_f1", 0.0)
    return f"Best model: {best} | accuracy={acc:.4f} | f1={f1:.4f}"
