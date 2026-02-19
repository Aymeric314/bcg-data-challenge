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
# Paths
# ---------------------------------------------------------------------------
HISTORICAL_PATH = GOLD_DATASETS_DIR / "historical.parquet"
PREDICTIONS_DIR = GOLD_DATASETS_DIR / "predictions"
FUTURE_SCENARIOS = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]

# Columns to drop before training
# 'year' is excluded because tree models can't extrapolate beyond
# the training range (1983-2014), making it misleading for 2015-2050 predictions.
DROP_COLS = ["department", "code_dep", "scenario", "year"]

# Hyperparameter distribution for randomized search
# Tuned for better generalization (reduced overfitting)
PARAM_DISTRIBUTION = {
    "max_depth": [2, 3, 4],  # Reduced from [3,4,5] to prevent deep trees
    "learning_rate": [0.01, 0.03, 0.05],  # Lower learning rates
    "min_child_weight": [10, 20, 30],  # Increased to prevent overfitting
    "gamma": [0, 0.1, 0.5],  # Minimum loss reduction for splits
    "subsample": [0.6, 0.7, 0.8],  # More aggressive subsampling
    "colsample_bytree": [0.5, 0.6, 0.7],  # More feature subsampling
    "reg_alpha": [0.5, 1.0, 2.0],  # Higher L1 regularization
    "reg_lambda": [5.0, 10.0, 20.0],  # Higher L2 regularization
}

# Number of random combinations to try (much faster than full grid)
N_ITER = 50


def train_xgboost():
    """Load historical data, tune hyperparameters, train and evaluate XGBoost."""

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

    logger.info(f"Features: {X.shape[1]} (year excluded — can't extrapolate beyond training range)")

    # --- Train/test split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Train: {X_train.shape[0]:,} samples, Test: {X_test.shape[0]:,} samples")

    # --- Hyperparameter tuning with cross-validation ----------------------
    # Calculate total combinations for reference
    total_combinations = 1
    for values in PARAM_DISTRIBUTION.values():
        total_combinations *= len(values)
    logger.info(
        f"Starting hyperparameter tuning: "
        f"trying {N_ITER} random combinations "
        f"(out of {total_combinations:,} total possible)"
    )
    logger.info(f"Using 5-fold CV, so {N_ITER * 5} model fits total")

    # XGBoost ≥ 2.0 uses device="cuda" instead of tree_method="gpu_hist"
    tree_method = "hist"
    device = "cpu"
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().split("\n")[0]
            logger.info(f"GPU detected: {gpu_name}. Using GPU acceleration.")
            device = "cuda"
        else:
            logger.info("No GPU detected. Using CPU.")
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        logger.info("GPU check failed. Using CPU.")

    # Create base model without early stopping (requires eval_set which causes data leakage)
    base_model = XGBRegressor(
        n_estimators=500,
        random_state=42,
        tree_method=tree_method,
        device=device,
    )

    start_time = time.time()

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=PARAM_DISTRIBUTION,
        n_iter=N_ITER,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=2,  # Show detailed progress: 0=silent, 1=some, 2=detailed
    )

    logger.info("Fitting models (this may take several minutes)...")
    logger.info(
        "Progress will be shown below. Each 'Fitting' line represents "
        "one parameter combination × 5 CV folds = 5 model fits."
    )
    # NOTE: Do NOT pass eval_set with test data here - that would cause data leakage!
    # RandomizedSearchCV handles its own cross-validation internally.
    # We'll only use the test set for final evaluation after tuning.
    random_search.fit(X_train, y_train)

    elapsed = time.time() - start_time
    best_params = random_search.best_params_
    logger.info(f"Hyperparameter tuning completed in {elapsed:.1f}s")
    logger.info(f"Best CV R²: {random_search.best_score_:.4f}")
    logger.info(f"Best params: {best_params}")

    # --- Train final model with best params -------------------------------
    model = random_search.best_estimator_

    # --- Evaluate ---------------------------------------------------------
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

    # --- Feature importance (top 15) --------------------------------------
    importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(
        ascending=False
    )

    logger.info("Top 15 features by importance:")
    for feat_name, imp in importance.head(15).items():
        logger.info(f"  {feat_name:30s} {imp:.4f}")

    # --- Refit on ALL historical data for production ----------------------
    logger.info("Refitting on ALL historical data for production …")
    feature_cols = [c for c in df.columns if c not in DROP_COLS + [target]]
    X_all = df[feature_cols]
    y_all = df[target]

    final_model = XGBRegressor(
        n_estimators=500,
        random_state=42,
        tree_method=tree_method,
        device=device,
        **best_params,
    )
    final_model.fit(X_all, y_all)

    full_r2 = r2_score(y_all, final_model.predict(X_all))
    logger.info(f"Full refit R²: {full_r2:.4f}")

    # --- Generate future predictions (2015–2050) -------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in FUTURE_SCENARIOS:
        scenario_path = GOLD_DATASETS_DIR / f"{scenario}.parquet"
        logger.info(f"Predicting {scenario} from {scenario_path}")
        future_df = pd.read_parquet(scenario_path)

        X_future = future_df[feature_cols]
        future_df["predicted_yield"] = final_model.predict(X_future)

        out_path = PREDICTIONS_DIR / f"{scenario}_xgboost_predictions.parquet"
        future_df.to_parquet(out_path, index=False)
        logger.success(f"Saved {scenario} predictions ({future_df.shape}) to {out_path}")

        logger.info(
            f"  {scenario} predicted yield — "
            f"mean: {future_df['predicted_yield'].mean():.2f}, "
            f"std: {future_df['predicted_yield'].std():.2f}, "
            f"min: {future_df['predicted_yield'].min():.2f}, "
            f"max: {future_df['predicted_yield'].max():.2f}"
        )

    return final_model, feature_cols


if __name__ == "__main__":
    logger.info("Starting XGBoost training pipeline")
    train_xgboost()
    logger.info("Finished XGBoost training pipeline")
