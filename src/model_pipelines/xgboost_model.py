"""
XGBoost Climate-Only Model
==========================
Predicts barley yield using **only** climate and spatial features
(no year, no autoregressive lags).

Key design choices
------------------
1. **Year excluded** — tree models cannot extrapolate beyond the
   training range (1983-2014), so ``year`` would be misleading for
   2015-2050 predictions.  All temporal signal must come from the
   climate inputs themselves.

2. **Department / code_dep excluded** — avoids learning opaque
   department IDs.  Spatial variation is captured by latitude,
   longitude, and altitude instead.

3. **Randomised hyperparameter search** — 50 random samples from a
   regularised parameter grid, evaluated with 5-fold CV.

4. **Production refit** — after evaluation the final model is retrained
   on *all* historical data before generating future predictions.
"""

import sys
import time
from pathlib import Path

# Add project root to Python path (must be before project imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from constants.paths import GOLD_DATASETS_DIR  # noqa: E402
from utils.logger import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Paths & scenarios
# ---------------------------------------------------------------------------
HISTORICAL_PATH = GOLD_DATASETS_DIR / "historical.parquet"
PREDICTIONS_DIR = GOLD_DATASETS_DIR / "predictions"
FUTURE_SCENARIOS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]

# Columns to drop before training.
# ``year`` is excluded because tree models cannot extrapolate beyond the
# training range (1983-2014), making it misleading for 2015-2050 predictions.
DROP_COLS = ["department", "code_dep", "scenario", "year"]

TARGET = "yield"

# ---------------------------------------------------------------------------
# Hyperparameter search space (regularised to prevent overfitting)
# ---------------------------------------------------------------------------
PARAM_DISTRIBUTION = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "min_child_weight": [10, 20, 30],
    "gamma": [0, 0.1, 0.5],
    "subsample": [0.6, 0.7, 0.8],
    "colsample_bytree": [0.5, 0.6, 0.7],
    "reg_alpha": [0.5, 1.0, 2.0],
    "reg_lambda": [5.0, 10.0, 20.0],
}
N_ITER = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_device() -> tuple[str, str]:
    """Return ``(tree_method, device)`` for XGBoost ≥ 2.0.

    XGBoost 2.0+ replaced ``tree_method='gpu_hist'`` with
    ``tree_method='hist', device='cuda'``.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            logger.info(f"GPU detected: {gpu_name}. Using GPU acceleration.")
            return "hist", "cuda"
    except Exception:
        pass

    logger.info("No GPU detected. Using CPU.")
    return "hist", "cpu"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train_xgboost():
    """Load historical data, tune hyperparameters, train and evaluate XGBoost.

    Steps
    -----
    1. Load and clean historical data.
    2. Random train / test split (80 / 20).
    3. Hyperparameter tuning via ``RandomizedSearchCV`` (5-fold CV).
    4. Evaluate best model on held-out test set.
    5. Refit on all historical data (production model).
    6. Generate future predictions for each SSP scenario.

    Returns
    -------
    tuple[XGBRegressor, list[str]]
        The production model and the list of feature column names.
    """

    # --- 1. Load data -----------------------------------------------------
    logger.info(f"Loading historical data from {HISTORICAL_PATH}")
    df = pd.read_parquet(HISTORICAL_PATH)
    logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

    before = len(df)
    df = df.dropna(subset=[TARGET])
    logger.info(f"Dropped {before - len(df)} rows with missing yield")

    # --- 2. Separate features / target ------------------------------------
    X = df.drop(columns=DROP_COLS + [TARGET])
    y = df[TARGET]
    feature_cols = list(X.columns)
    logger.info(f"Features: {len(feature_cols)} (year excluded)")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    logger.info(f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

    # --- 3. Hyperparameter tuning -----------------------------------------
    total_combos = 1
    for values in PARAM_DISTRIBUTION.values():
        total_combos *= len(values)
    logger.info(
        f"Hyperparameter tuning: {N_ITER} random combos "
        f"(of {total_combos:,}) × 5 CV folds = {N_ITER * 5} fits"
    )

    tree_method, device = _detect_device()

    base_model = XGBRegressor(
        n_estimators=500,
        random_state=42,
        tree_method=tree_method,
        device=device,
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTION,
        n_iter=N_ITER,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    best_params = random_search.best_params_
    logger.info(f"Tuning completed in {elapsed:.1f}s")
    logger.info(f"Best CV R²: {random_search.best_score_:.4f}")
    logger.info(f"Best params: {best_params}")

    # --- 4. Evaluate on held-out test set ---------------------------------
    model = random_search.best_estimator_

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    logger.info(f"Train R²:  {train_r2:.4f}")
    logger.info(f"Test  R²:  {test_r2:.4f}")
    logger.info(f"Test  RMSE: {test_rmse:.4f}")
    logger.info(f"Test  MAE:  {test_mae:.4f}")

    # Feature importance (top 15)
    importance = pd.Series(
        model.feature_importances_,
        index=feature_cols,
    ).sort_values(ascending=False)

    logger.info("Top 15 features by importance:")
    for feat, imp in importance.head(15).items():
        logger.info(f"  {feat:30s} {imp:.4f}")

    # --- 5. Refit on ALL historical data ----------------------------------
    logger.info("Refitting on ALL historical data for production …")

    final_model = XGBRegressor(
        n_estimators=500,
        random_state=42,
        tree_method=tree_method,
        device=device,
        **best_params,
    )
    final_model.fit(df[feature_cols], df[TARGET])

    full_r2 = r2_score(df[TARGET], final_model.predict(df[feature_cols]))
    logger.info(f"Full refit R² (in-sample): {full_r2:.4f}")

    # --- 6. Generate future predictions (2015-2050) -----------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in FUTURE_SCENARIOS:
        scenario_path = GOLD_DATASETS_DIR / f"{scenario}.parquet"
        logger.info(f"Predicting {scenario} from {scenario_path}")
        future_df = pd.read_parquet(scenario_path)

        future_df["predicted_yield"] = final_model.predict(
            future_df[feature_cols],
        )

        out_path = PREDICTIONS_DIR / f"{scenario}_xgboost_predictions.parquet"
        future_df.to_parquet(out_path, index=False)
        logger.success(f"Saved {scenario} predictions ({future_df.shape}) to {out_path}")

        pred = future_df["predicted_yield"]
        logger.info(
            f"  {scenario} yield — mean: {pred.mean():.2f}, "
            f"std: {pred.std():.2f}, "
            f"min: {pred.min():.2f}, max: {pred.max():.2f}"
        )

    return final_model, feature_cols


if __name__ == "__main__":
    logger.info("Starting XGBoost training pipeline")
    train_xgboost()
    logger.info("Finished XGBoost training pipeline")
