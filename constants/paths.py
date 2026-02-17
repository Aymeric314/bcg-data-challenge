"""File paths constants for the project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Config directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DATASETS_DIR = DATA_DIR / "bronze_datasets"
SILVER_DATASETS_DIR = DATA_DIR / "silver_datasets"
GOLD_DATASETS_DIR = DATA_DIR / "gold_datasets"

# Data files
BARLEY_YIELD_CSV = BRONZE_DATASETS_DIR / "barley_yield_from_1982.csv"
CLIMATE_DATA_PARQUET = BRONZE_DATASETS_DIR / "climate_data_from_1982.parquet"
DEPARTMENT_MAP_CSV = BRONZE_DATASETS_DIR / "df_map_mean_yield.csv"
COMMUNES_CSV = BRONZE_DATASETS_DIR / "communes-france-2025.csv"
