import sys
from pathlib import Path

# Add project root to Python path (must be before project imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from constants.paths import GOLD_DATASETS_DIR  # noqa: E402
from utils.logger import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HISTORICAL_PATH = GOLD_DATASETS_DIR / "historical.parquet"

# Columns to drop before training
DROP_COLS = ["department", "code_dep", "scenario", "year"]


def train_xgboost():
    """Load historical data, train/test split, and train an XGBoost model."""

    # --- Load data --------------------------------------------------------
    logger.info(f"Loading historical data from {HISTORICAL_PATH}")
    df = pd.read_parquet(HISTORICAL_PATH)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    # --- Drop rows without yield ------------------------------------------
    before = len(df)
    df = df.dropna(subset=["yield"])
    logger.info(f"Dropped {before - len(df)} rows with missing yield")

    # --- Separate features and target -------------------------------------
    target = "yield"
    X = df.drop(columns=DROP_COLS + [target])
    y = df[target]

    logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")

    # --- Train/test split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

    # --- Train XGBoost ----------------------------------------------------
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=50,
    )

    logger.info("Training XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --- Evaluate ---------------------------------------------------------
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    logger.info(f"Train R²: {train_score:.4f}")
    logger.info(f"Test  R²: {test_score:.4f}")

    # --- Feature importance (top 15) --------------------------------------
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    logger.info("Top 15 features by importance:")
    for feat_name, imp in importance.head(15).items():
        logger.info(f"  {feat_name:30s} {imp:.4f}")

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    logger.info("Starting XGBoost training pipeline")
    train_xgboost()
    logger.info("Finished XGBoost training pipeline")

