"""File paths constants for the project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DATASETS_DIR = DATA_DIR / "bronze_datasets"
SILVER_DATASETS_DIR = DATA_DIR / "silver_datasets"

# Data files
BARLEY_YIELD_CSV = BRONZE_DATASETS_DIR / "barley_yield_from_1982.csv"
CLIMATE_DATA_PARQUET = BRONZE_DATASETS_DIR / "climate_data_from_1982.parquet"
