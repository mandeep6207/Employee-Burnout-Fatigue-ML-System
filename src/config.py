from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "wfh_burnout_dataset.csv"
STANDARD_DATA_PATH = DATA_DIR / "burnout_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
VISUALS_DIR = PROJECT_ROOT / "visuals"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = PROJECT_ROOT / "metrics"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET_COLUMN = "burnout_risk"
TARGET_LABEL_ORDER = ["Low", "Medium", "High"]
CATEGORICAL_COLUMNS = ["day_type"]
RAW_NUMERIC_COLUMNS = [
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
ENGINEERED_COLUMNS = [
    "productivity_ratio",
    "workload_index",
    "fatigue_level",
    "sleep_efficiency",
    "work_life_balance_score",
]
MODEL_FEATURE_COLUMNS = RAW_NUMERIC_COLUMNS + ENGINEERED_COLUMNS + CATEGORICAL_COLUMNS
LEAKAGE_COLUMNS = ["burnout_score", "fatigue_score", "user_id"]
