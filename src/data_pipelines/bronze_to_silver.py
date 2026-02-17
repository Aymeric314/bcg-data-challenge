import sys
from pathlib import Path

# Add project root to Python path (must be before project imports)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from constants.paths import (  # noqa: E402
    BARLEY_YIELD_CSV,
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

    # 2. Process barley yield data (fill missing values)
    logger.info("Processing barley yield data")
    processed_barley_df = process_barley_yield_data(barley_df)

    # 3. Write processed data to silver
    silver_barley_path = SILVER_DATASETS_DIR / "barley_yield_processed.parquet"
    logger.info(f"Writing processed barley yield data to {silver_barley_path}")
    processed_barley_df.to_parquet(silver_barley_path, index=False)

    logger.success(f"Barley yield data processing complete. Shape: {processed_barley_df.shape}")


if __name__ == "__main__":
    # Execute the pipeline --> Called using DVC
    logger.info("Starting Bronze to Silver pipeline")
    bronze_to_silver()
    logger.info("Finished Bronze to Silver pipeline")
