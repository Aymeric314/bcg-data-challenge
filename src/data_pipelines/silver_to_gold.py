import sys
from pathlib import Path

# Add project root to Python path (must be before project imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from constants.paths import (  # noqa: E402
    DEPARTMENT_MAP_CSV,
    GOLD_DATASETS_DIR,
    SILVER_DATASETS_DIR,
)
from utils.logger import logger  # noqa: E402

# ---------------------------------------------------------------------------
# Silver dataset paths
# ---------------------------------------------------------------------------
SILVER_CLIMATE_PATH = SILVER_DATASETS_DIR / "climate_data_processed.parquet"
SILVER_BARLEY_PATH = SILVER_DATASETS_DIR / "barley_yield_processed.parquet"
SILVER_ALTITUDE_PATH = SILVER_DATASETS_DIR / "department_altitude.parquet"

# ---------------------------------------------------------------------------
# Growing-season window definitions
# Each window maps to months in the *previous* calendar year (prev_year)
# and months in the *harvest* year.
# Harvest year Y  ⟹  season runs Oct 1 (Y-1) → Jun 30 (Y).
# ---------------------------------------------------------------------------
WINDOWS = {
    "ON": {"prev_year": [10, 11], "harvest_year": []},
    "DJF": {"prev_year": [12], "harvest_year": [1, 2]},
    "FM": {"prev_year": [], "harvest_year": [2, 3]},
    "MA": {"prev_year": [], "harvest_year": [3, 4]},
    "MJ": {"prev_year": [], "harvest_year": [5, 6]},
    "SEASON": {
        "prev_year": [10, 11, 12],
        "harvest_year": [1, 2, 3, 4, 5, 6],
    },
}

# Windows where heat-stress features (Tmax) are computed
HEAT_WINDOWS = {"MA", "MJ", "SEASON"}

# Windows where cold-stress features are computed
COLD_WINDOWS = {"DJF"}


# ===================================================================
# Helper functions
# ===================================================================


def _max_consecutive_dry_days(precip_values, threshold=1.0):
    """Maximum consecutive dry days where P < *threshold* mm/day.

    Uses a vectorised boundary-diff trick (no Python loop).

    Args:
        precip_values: 1-D numpy array of daily precipitation (mm/day).
        threshold: Dry-day threshold in mm/day (default 1.0).

    Returns:
        int – longest run of consecutive dry days.
    """
    if len(precip_values) == 0:
        return 0
    is_dry = precip_values < threshold
    if not is_dry.any():
        return 0
    # Pad with False so transitions are detected at edges
    padded = np.concatenate(([False], is_dry, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return int((ends - starts).max())


def _compute_window_features(tmean, tmax, precip, window_name):
    """Compute all climate features for one time window.

    Args:
        tmean: numpy array – daily mean temperature (°C).
        tmax:  numpy array – daily max temperature (°C).
        precip: numpy array – daily precipitation (mm/day).
        window_name: str – window label (ON, DJF, FM, MA, MJ, SEASON).

    Returns:
        dict of {feature_name: value}.
    """
    feat = {}
    n = len(tmean)

    # --- 1. Temperature development (Tmean) — all windows --------
    W = window_name
    feat[f"tmean_mean_{W}"] = float(tmean.mean()) if n > 0 else np.nan
    feat[f"tmean_std_{W}"] = float(tmean.std(ddof=1)) if n > 1 else np.nan
    feat[f"gdd0_sum_{W}"] = float(np.maximum(tmean, 0).sum()) if n > 0 else 0.0

    # Cold risk — DJF only
    if W in COLD_WINDOWS:
        feat[f"cold_days_0_{W}"] = int((tmean <= 0).sum()) if n > 0 else 0
        feat[f"cold_days_m2_{W}"] = int((tmean <= -2).sum()) if n > 0 else 0

    # --- 2. Heat stress (Tmax) — MA, MJ, SEASON ------------------
    if W in HEAT_WINDOWS:
        feat[f"tmax_mean_{W}"] = float(tmax.mean()) if n > 0 else np.nan
        feat[f"hot_days_25_{W}"] = int((tmax >= 25).sum()) if n > 0 else 0
        feat[f"hot_days_30_{W}"] = int((tmax >= 30).sum()) if n > 0 else 0
        feat[f"tmax_p95_{W}"] = float(np.percentile(tmax, 95)) if n > 0 else np.nan

    # --- 3. Water availability (P) — all windows -----------------
    feat[f"ppt_sum_{W}"] = float(precip.sum()) if n > 0 else 0.0
    feat[f"rain_days_1_{W}"] = int((precip >= 1).sum()) if n > 0 else 0
    feat[f"heavy_rain_10_{W}"] = int((precip >= 10).sum()) if n > 0 else 0
    feat[f"dry_days_1_{W}"] = int((precip < 1).sum()) if n > 0 else 0
    feat[f"max_cdd_1_{W}"] = _max_consecutive_dry_days(precip)

    return feat


# ===================================================================
# Main aggregation function
# ===================================================================


def aggregate_climate_features(
    climate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate daily climate data into growing-season features.

    Produces one row per (department, scenario, harvest_year) with 64
    climate features computed over six growing-season windows:
        ON, DJF, FM, MA, MJ, SEASON.

    Growing season: Oct 1 (Y-1) → Jun 30 (Y), where Y = harvest year.

    Transition rule (historical → future):
        For harvest year 2015 the Oct–Dec 2014 portion comes from the
        historical scenario; Jan–Jun 2015 onward comes from each SSP
        scenario.

    Unit conversions applied automatically:
        - Temperature : Kelvin → °C
        - Precipitation: kg m⁻² s⁻¹ → mm day⁻¹

    Args:
        climate_df: Silver climate dataframe (daily rows).

    Returns:
        DataFrame with columns [department, scenario, year, <64 features>].
    """
    df = climate_df.copy()

    # --- Unit conversions -----------------------------------------
    logger.info("Converting units: K → °C, kg/m²/s → mm/day")
    df["tmean"] = df["near_surface_air_temperature"] - 273.15
    df["tmax"] = df["daily_maximum_near_surface_air_temperature"] - 273.15
    df["precip"] = df["precipitation"] * 86400

    # --- Time parsing & growing-season filter ---------------------
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["cal_year"] = df["time"].dt.year

    # Keep only Oct–Jun (growing season); drop Jul–Sep
    growing_months = [10, 11, 12, 1, 2, 3, 4, 5, 6]
    df = df[df["month"].isin(growing_months)].copy()

    # Assign harvest year:
    #   Oct–Dec → harvest_year = cal_year + 1
    #   Jan–Jun → harvest_year = cal_year
    df["harvest_year"] = np.where(
        df["month"].isin([10, 11, 12]),
        df["cal_year"] + 1,
        df["cal_year"],
    )

    # --- Historical → future transition --------------------------
    future_scenarios = sorted(s for s in df["scenario"].unique() if s != "historical")
    hist_transition = df[
        (df["scenario"] == "historical")
        & (df["cal_year"] == 2014)
        & (df["month"].isin([10, 11, 12]))
    ]
    n_trans = len(hist_transition)
    n_future = len(future_scenarios)
    logger.info(
        f"Transition: copying {n_trans} historical rows "
        f"(Oct–Dec 2014) into {n_future} future scenarios"
    )
    parts = []
    for scenario in future_scenarios:
        part = hist_transition.copy()
        part["scenario"] = scenario
        parts.append(part)
    if parts:
        df = pd.concat([df] + parts, ignore_index=True)

    # --- Keep only complete harvest years -------------------------
    # Historical: 1983–2014  (needs Oct Y-1 … Jun Y)
    # Future:     2015–2050
    valid = ((df["scenario"] == "historical") & df["harvest_year"].between(1983, 2014)) | (
        (df["scenario"] != "historical") & df["harvest_year"].between(2015, 2050)
    )
    df = df[valid].copy()

    # Sort for consecutive-day calculations
    df = df.sort_values(["nom_dep", "scenario", "harvest_year", "time"])

    # --- Compute features per group -------------------------------
    groups = df.groupby(["nom_dep", "code_dep", "scenario", "harvest_year"])
    n_groups = len(groups)
    logger.info(f"Computing features for {n_groups:,} groups …")

    results = []
    for i, ((dept, code, scenario, hyear), gdf) in enumerate(groups):
        if (i + 1) % 2000 == 0 or i == 0:
            logger.info(f"  Progress: {i + 1:,} / {n_groups:,}")

        month_arr = gdf["month"].values
        cyear_arr = gdf["cal_year"].values
        tmean_arr = gdf["tmean"].values
        tmax_arr = gdf["tmax"].values
        precip_arr = gdf["precip"].values

        row = {
            "department": dept,
            "code_dep": code,
            "scenario": scenario,
            "year": int(hyear),
        }

        # Compute features for each window
        for wname, wdef in WINDOWS.items():
            mask = np.zeros(len(gdf), dtype=bool)
            if wdef["prev_year"]:
                mask |= np.isin(month_arr, wdef["prev_year"]) & (cyear_arr == hyear - 1)
            if wdef["harvest_year"]:
                mask |= np.isin(month_arr, wdef["harvest_year"]) & (cyear_arr == hyear)

            row.update(
                _compute_window_features(
                    tmean_arr[mask],
                    tmax_arr[mask],
                    precip_arr[mask],
                    wname,
                )
            )

        # --- 4. Balance / interaction features --------------------
        row["gdd0_per_ppt_SEASON"] = row["gdd0_sum_SEASON"] / (row["ppt_sum_SEASON"] + 1)
        row["heat30_per_ppt_MJ"] = row["hot_days_30_MJ"] / (row["ppt_sum_MJ"] + 1)

        results.append(row)

    result_df = pd.DataFrame(results)
    n_feat = len(result_df.columns) - 4  # minus key columns
    logger.info(f"Climate features computed: {result_df.shape[0]:,} rows × {n_feat} features")
    return result_df


# ===================================================================
# Department altitude enrichment
# ===================================================================


def add_department_altitude(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add altitude column by matching department code.

    Reads department altitude from the silver altitude parquet and
    merges it onto the climate features dataframe using code_dep.

    Args:
        df: Climate features dataframe with 'code_dep' column.

    Returns:
        DataFrame with 'altitude' column added.
    """
    alt_df = pd.read_parquet(SILVER_ALTITUDE_PATH)
    alt_merge = alt_df[["code", "altitude_moyenne"]].rename(
        columns={"code": "code_dep", "altitude_moyenne": "altitude"}
    )

    initial_rows = len(df)
    merged = df.merge(alt_merge, on="code_dep", how="left")

    missing = merged["altitude"].isna().sum()
    if missing > 0:
        unmatched = merged.loc[merged["altitude"].isna(), "code_dep"].unique()
        logger.warning(f"{missing} rows missing altitude. Unmatched code_dep: {list(unmatched)}")
    else:
        logger.info(f"Added altitude for all {initial_rows:,} rows")

    return merged


# ===================================================================
# Department coordinate enrichment
# ===================================================================


def add_department_coordinates(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add latitude and longitude columns by matching department name.

    Reads department centroids from the map CSV and merges them onto
    the climate features dataframe.

    Args:
        df: Climate features dataframe with 'department' column.

    Returns:
        DataFrame with 'latitude' and 'longitude' columns added.
    """
    # Load department coordinate mapping
    dept_map = pd.read_csv(DEPARTMENT_MAP_CSV)
    dept_coords = dept_map[["department", "lat", "lon"]].rename(
        columns={"lat": "latitude", "lon": "longitude"}
    )

    initial_rows = len(df)
    merged = df.merge(dept_coords, on="department", how="left")

    # Validate
    missing = merged["latitude"].isna().sum()
    if missing > 0:
        unmatched = merged.loc[merged["latitude"].isna(), "department"].unique()
        logger.warning(
            f"{missing} rows missing coordinates. Unmatched departments: {list(unmatched)}"
        )
    else:
        logger.info(f"Added lat/lon for all {initial_rows:,} rows")

    return merged


# ===================================================================
# Split by scenario & attach yield
# ===================================================================


def split_and_save_by_scenario(
    climate_features: pd.DataFrame,
) -> None:
    """Split climate features by scenario, attach yield to historical.

    Produces four gold datasets:
        - historical.parquet   (with yield, area, production from barley)
        - ssp1_2_6.parquet
        - ssp2_4_5.parquet
        - ssp5_8_5.parquet

    Args:
        climate_features: Full climate features dataframe with all
            scenarios.
    """
    # Load barley yield data
    barley_df = pd.read_parquet(SILVER_BARLEY_PATH)

    # Drop unnamed index column if present
    barley_df = barley_df.drop(
        columns=[c for c in barley_df.columns if "Unnamed" in c],
        errors="ignore",
    )

    # --- Historical: merge with barley yield ----------------------
    hist_df = climate_features[climate_features["scenario"] == "historical"].copy()

    hist_merged = hist_df.merge(
        barley_df[["department", "year", "yield"]],
        on=["department", "year"],
        how="left",
    )

    matched = hist_merged["yield"].notna().sum()
    total = len(hist_merged)
    logger.info(f"Historical: {matched}/{total} rows matched with yield")

    hist_path = GOLD_DATASETS_DIR / "historical.parquet"
    hist_merged.to_parquet(hist_path, index=False)
    logger.success(f"Saved historical dataset ({hist_merged.shape}) to {hist_path}")

    # --- Future scenarios: save individually ----------------------
    future_scenarios = ["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"]
    for scenario in future_scenarios:
        scenario_df = climate_features[climate_features["scenario"] == scenario].copy()

        scenario_path = GOLD_DATASETS_DIR / f"{scenario}.parquet"
        scenario_df.to_parquet(scenario_path, index=False)
        logger.success(f"Saved {scenario} dataset ({scenario_df.shape}) to {scenario_path}")


# ===================================================================
# Pipeline orchestration
# ===================================================================


def silver_to_gold():
    """Pipeline to transform silver data to gold data.

    This function uses the previously defined functions to read silver
    data, process it, and write the gold data.

    To add a step, define the function above and call it here.
    """
    # Ensure gold directory exists
    GOLD_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Climate feature aggregation ------------------------------
    logger.info(f"Reading climate data from {SILVER_CLIMATE_PATH}")
    climate_df = pd.read_parquet(SILVER_CLIMATE_PATH)

    logger.info("Aggregating climate features by growing season")
    climate_features = aggregate_climate_features(climate_df)

    logger.info("Adding department coordinates (lat/lon)")
    climate_features = add_department_coordinates(climate_features)

    logger.info("Adding department altitude")
    climate_features = add_department_altitude(climate_features)

    logger.info(f"Full climate features shape: {climate_features.shape}")

    # --- Split by scenario and save -------------------------------
    logger.info("Splitting by scenario and attaching yield")
    split_and_save_by_scenario(climate_features)


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    logger.info("Starting Silver to Gold pipeline")
    silver_to_gold()
    logger.info("Finished Silver to Gold pipeline")
