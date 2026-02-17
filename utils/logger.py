"""Centralized logging configuration using Loguru.

This module provides a simple console logger that can be imported and used
across the project. Configuration is loaded from config/loguru.yaml.

Usage:
    from src.utils.logger import logger

    logger.info("Processing started")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")
    logger.success("Operation completed successfully")
"""

import sys

import yaml
from loguru import logger

from constants.paths import CONFIG_DIR


def setup_logger():
    """Set up logger from YAML configuration file."""
    # Remove default handler
    logger.remove()

    # Load configuration from YAML
    config_path = CONFIG_DIR / "loguru.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Add console handler from config
    console_config = config["handlers"]["console"]

    logger.add(
        sys.stderr,
        format=console_config["format"],
        level=console_config["level"],
        colorize=console_config["colorize"],
    )


# Initialize logger on import
setup_logger()

__all__ = ["logger"]
