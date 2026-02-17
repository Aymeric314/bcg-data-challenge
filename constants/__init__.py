"""Constants package."""

from .departments import DEPARTMENTS_TO_EXCLUDE
from .paths import (
    BARLEY_YIELD_CSV,
    CLIMATE_DATA_PARQUET,
    DATA_DIR,
    PROJECT_ROOT,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "BARLEY_YIELD_CSV",
    "CLIMATE_DATA_PARQUET",
    "DEPARTMENTS_TO_EXCLUDE",
]
