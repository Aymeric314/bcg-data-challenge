import sys
from pathlib import Path

# Add project root to Python path (must be before project imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from constants.departments import DEPARTMENTS_TO_EXCLUDE  # noqa: E402
from constants.paths import (  # noqa: E402
    BARLEY_YIELD_CSV,
    CLIMATE_DATA_PARQUET,
    COMMUNES_CSV,
    SILVER_DATASETS_DIR,
)
from utils.logger import logger  # noqa: E402


def process_barley_yield_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process barley yield data by filling missing values.

    Fills missing values using the relationships:
    - If yield is missing but area and production are available:
      yield = round(production / area, 5)
    - If production is missing but yield and area are available:
      production = yield × area

    Args:
        df: Raw barley yield dataframe

    Returns:
        Processed dataframe with missing values filled
    """
    df = df.copy()

    # Calculate yield where it's missing but area and production are available
    missing_yield_mask = df["yield"].isna() & df["area"].notna() & df["production"].notna()
    if missing_yield_mask.sum() > 0:
        logger.info(
            f"Calculating {missing_yield_mask.sum()} missing yield values from area and production"
        )
        df.loc[missing_yield_mask, "yield"] = (
            df.loc[missing_yield_mask, "production"] / df.loc[missing_yield_mask, "area"]
        ).round(5)

    # Calculate production where it's missing but yield and area are available
    missing_production_mask = df["production"].isna() & df["yield"].notna() & df["area"].notna()
    if missing_production_mask.sum() > 0:
        logger.info(
            f"Calculating {missing_production_mask.sum()} missing production from yield and area"
        )
        df.loc[missing_production_mask, "production"] = (
            df.loc[missing_production_mask, "yield"] * df.loc[missing_production_mask, "area"]
        )

    # Log remaining missing values
    remaining_missing = df[["yield", "area", "production"]].isna().sum()
    if remaining_missing.sum() > 0:
        logger.warning(f"Remaining missing values after calculation: {remaining_missing.to_dict()}")
    else:
        logger.info("All missing values have been filled")

    return df


def normalize_barley_department_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize department names in barley dataset.

    Strips whitespace and converts to lowercase for consistent matching.

    Args:
        df: Barley yield dataframe with 'department' column

    Returns:
        Dataframe with normalized department names
    """
    df = df.copy()
    if "department" in df.columns:
        initial_unique = df["department"].nunique()
        df["department"] = df["department"].str.strip().str.lower()
        final_unique = df["department"].nunique()
        logger.info(
            f"Normalized barley department names. "
            f"Unique departments: {initial_unique} -> {final_unique}"
        )
    else:
        logger.warning("'department' column not found in barley dataframe")
    return df


def remove_excluded_departments(df: pd.DataFrame) -> pd.DataFrame:
    """Remove departments that are not present in climate data.

    Args:
        df: Barley yield dataframe with normalized 'department' column

    Returns:
        Dataframe with excluded departments removed
    """
    df = df.copy()
    if "department" not in df.columns:
        logger.warning("'department' column not found in dataframe")
        return df

    initial_shape = df.shape[0]
    initial_departments = df["department"].nunique()

    # Remove rows with excluded departments
    df_filtered = df[~df["department"].isin(DEPARTMENTS_TO_EXCLUDE)].copy()

    removed_count = initial_shape - df_filtered.shape[0]
    final_departments = df_filtered["department"].nunique()

    if removed_count > 0:
        num_excluded = len(DEPARTMENTS_TO_EXCLUDE)
        logger.info(
            f"Removed {removed_count} rows from {num_excluded} excluded departments. "
            f"Departments: {initial_departments} -> {final_departments}"
        )
        logger.info(f"Excluded departments: {DEPARTMENTS_TO_EXCLUDE}")
    else:
        logger.info("No rows from excluded departments found")

    return df_filtered


def remove_invalid_production_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with 0 or NaN for production.

    Args:
        df: Dataframe with production column

    Returns:
        Dataframe with rows having 0 or NaN production removed
    """
    initial_shape = df.shape[0]
    df_filtered = df[(df["production"].notna()) & (df["production"] != 0)].copy()
    removed_count = initial_shape - df_filtered.shape[0]
    if removed_count > 0:
        logger.info(f"Removed {removed_count} rows with 0 or NaN production.")
    else:
        logger.info("No rows with 0 or NaN production found")
    return df_filtered


def process_climate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process climate data by pivoting metric column into separate columns.

    Merges rows by matching on all columns except 'metric' and 'value', then
    converts the 'metric' column values into new column names with 'value' as
    the corresponding values. The original 'metric' and 'value' columns are
    dropped after pivoting.

    Args:
        df: Raw climate dataframe with 'metric' and 'value' columns

    Returns:
        Processed dataframe with metric values as columns
    """
    df = df.copy()

    # Identify index columns (all columns except 'metric' and 'value')
    index_cols = [col for col in df.columns if col not in ["metric", "value"]]

    # Log the pivot operation
    unique_metrics = df["metric"].unique()
    logger.info(
        f"Pivoting climate data: {len(unique_metrics)} metrics found: {list(unique_metrics)}"
    )
    logger.info(f"Using {len(index_cols)} index columns: {index_cols}")

    # Pivot the dataframe
    df_pivoted = df.pivot_table(
        index=index_cols,
        columns="metric",
        values="value",
        aggfunc="first",  # Use first in case of duplicates
    ).reset_index()

    # Flatten column names if multi-index
    if isinstance(df_pivoted.columns, pd.MultiIndex):
        df_pivoted.columns = [col[0] if col[1] == "" else col[1] for col in df_pivoted.columns]
    else:
        # Rename columns to remove 'metric' level if present
        df_pivoted.columns.name = None

    logger.info(f"Climate data pivoted. Shape: {df.shape} -> {df_pivoted.shape}")

    return df_pivoted


def normalize_climate_department_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize department names in climate dataset.

    Strips whitespace and converts to lowercase for consistent matching.

    Args:
        df: Climate dataframe with 'nom_dep' column

    Returns:
        Dataframe with normalized department names
    """
    df = df.copy()
    if "nom_dep" in df.columns:
        initial_unique = df["nom_dep"].nunique()
        df["nom_dep"] = df["nom_dep"].str.strip().str.lower()
        final_unique = df["nom_dep"].nunique()
        logger.info(
            f"Normalized climate department names. "
            f"Unique departments: {initial_unique} -> {final_unique}"
        )
    else:
        logger.warning("'nom_dep' column not found in climate dataframe")
    return df


def remove_rows_with_nan_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN values in any metric column.

    Args:
        df: Processed climate dataframe with metric columns

    Returns:
        Dataframe with rows having NaN in any metric column removed
    """
    # Define metric columns
    metric_columns = [
        "daily_maximum_near_surface_air_temperature",
        "near_surface_air_temperature",
        "precipitation",
    ]

    # Check which metric columns exist in the dataframe
    existing_metric_cols = [col for col in metric_columns if col in df.columns]

    if not existing_metric_cols:
        logger.warning("No metric columns found in dataframe")
        return df

    initial_shape = df.shape[0]

    # Remove rows with NaN in any metric column
    df_filtered = df.dropna(subset=existing_metric_cols).copy()

    removed_count = initial_shape - df_filtered.shape[0]
    if removed_count > 0:
        remaining_rows = df_filtered.shape[0]
        logger.info(
            f"Removed {removed_count} rows with NaN in metric columns. "
            f"Remaining rows: {remaining_rows}"
        )
    else:
        logger.info("No rows with NaN in metric columns found")

    return df_filtered


def compute_department_altitude(df: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted average altitude per department from communes data.

    Extracts the 2-character department code from code_insee, then computes
    the weighted average of altitude_moyenne using superficie_km2 as weight.
    Also includes a normalized department name for matching with other datasets.

    Args:
        df: Communes dataframe with code_insee, dep_code, dep_nom,
            altitude_moyenne, superficie_km2

    Returns:
        Dataframe with columns: code, department, altitude_moyenne
    """
    import unicodedata

    df = df.copy()

    # Extract department code (first 2 characters of code_insee)
    df["code"] = df["code_insee"].astype(str).str[:2]

    # Drop rows where altitude or surface area is missing
    df = df.dropna(subset=["altitude_moyenne", "superficie_km2"])
    df = df[df["superficie_km2"] > 0]

    # Weighted average: sum(altitude * surface) / sum(surface)
    def weighted_avg(group: pd.DataFrame) -> float:
        return (group["altitude_moyenne"] * group["superficie_km2"]).sum() / group[
            "superficie_km2"
        ].sum()

    result = df.groupby("code").apply(weighted_avg, include_groups=False).reset_index()
    result.columns = ["code", "altitude_moyenne"]
    result["altitude_moyenne"] = result["altitude_moyenne"].round(2)

    # Build code → normalized department name mapping from dep_code/dep_nom
    name_map = df[["code", "dep_nom"]].drop_duplicates(subset=["code"]).sort_values("code")

    def _normalize_dep_name(name: str) -> str:
        """Normalize French department name to match climate data format."""
        # Remove accents
        nfkd = unicodedata.normalize("NFKD", name)
        ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
        # Lowercase, replace hyphens/spaces/apostrophes with underscore
        return ascii_name.strip().lower().replace("-", "_").replace("'", "_").replace(" ", "_")

    name_map["department"] = name_map["dep_nom"].apply(_normalize_dep_name)
    result = result.merge(name_map[["code", "department"]], on="code", how="left")

    logger.info(f"Computed weighted avg altitude for {len(result)} departments")

    return result[["code", "department", "altitude_moyenne"]]


def bronze_to_silver():
    """Pipeline to transform bronze data to silver data.

    This function uses the previously defined functions to read bronze data,
    process it, and write the silver data.

    To add a step to this pipeline, define the function above and call it here.
    """
    # The structure is always the same:

    # 1. Read data from bronze:
    # logger.info(f"Reading XXXX data at {PATH_TO_DATA}")
    # data = pd.read_csv(PATH_TO_DATA)

    # 2. Process data
    # logger.info("Processing XXXX data")
    # processed_data = process_xxxx(data) -> Use your function here

    # 3. Write data to silver
    # logger.info(f"Writing processed data to {PATH_TO_SILVER}")
    # processed_data.to_parquet(PATH_TO_SILVER)

    # Ensure silver directory exists
    SILVER_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read barley yield data from bronze
    logger.info(f"Reading barley yield data from {BARLEY_YIELD_CSV}")
    barley_df = pd.read_csv(BARLEY_YIELD_CSV, sep=";")

    # 2. Normalize department names
    logger.info("Normalizing barley department names")
    barley_df = normalize_barley_department_names(barley_df)

    # 3. Remove excluded departments (not in climate data)
    logger.info("Removing excluded departments")
    barley_df = remove_excluded_departments(barley_df)

    # 4. Process barley yield data (fill missing values)
    logger.info("Processing barley yield data")
    processed_barley_df = process_barley_yield_data(barley_df)

    # 6. Remove rows with 0 or NaN for production
    logger.info("Removing rows with 0 or NaN production")
    processed_barley_df = remove_invalid_production_rows(processed_barley_df)

    # 7. Write processed data to silver
    silver_barley_path = SILVER_DATASETS_DIR / "barley_yield_processed.parquet"
    logger.info(f"Writing processed barley yield data to {silver_barley_path}")
    processed_barley_df.to_parquet(silver_barley_path, index=False)

    logger.success(f"Barley yield data processing complete. Shape: {processed_barley_df.shape}")

    # 5. Read climate data from bronze
    logger.info(f"Reading climate data from {CLIMATE_DATA_PARQUET}")
    climate_df = pd.read_parquet(CLIMATE_DATA_PARQUET)

    # 6. Normalize department names
    logger.info("Normalizing climate department names")
    climate_df = normalize_climate_department_names(climate_df)

    # 7. Process climate data (pivot metric column)
    logger.info("Processing climate data")
    processed_climate_df = process_climate_data(climate_df)

    # 8. Remove rows with NaN in any metric column
    logger.info("Removing rows with NaN in metric columns")
    processed_climate_df = remove_rows_with_nan_metrics(processed_climate_df)

    # 9. Write processed climate data to silver
    silver_climate_path = SILVER_DATASETS_DIR / "climate_data_processed.parquet"
    logger.info(f"Writing processed climate data to {silver_climate_path}")
    processed_climate_df.to_parquet(silver_climate_path, index=False)

    logger.success(f"Climate data processing complete. Shape: {processed_climate_df.shape}")

    # 10. Read communes data from bronze
    logger.info(f"Reading communes data from {COMMUNES_CSV}")
    communes_df = pd.read_csv(COMMUNES_CSV, low_memory=False)

    # 11. Compute weighted average altitude per department
    logger.info("Computing weighted average altitude per department")
    dept_altitude_df = compute_department_altitude(communes_df)

    # 12. Write department altitude data to silver
    silver_altitude_path = SILVER_DATASETS_DIR / "department_altitude.parquet"
    logger.info(f"Writing department altitude data to {silver_altitude_path}")
    dept_altitude_df.to_parquet(silver_altitude_path, index=False)

    logger.success(f"Department altitude processing complete. Shape: {dept_altitude_df.shape}")


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    logger.info("Starting Bronze to Silver pipeline")
    bronze_to_silver()
    logger.info("Finished Bronze to Silver pipeline")
