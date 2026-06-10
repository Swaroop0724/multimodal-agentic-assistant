"""
config/logging_config.py
Structured logging via loguru. Call setup_logging() once at app entry point.
"""

import sys
from loguru import logger
from config.settings import settings


def setup_logging():
    """Configure loguru for the whole project."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level,
        colorize=True,
    )

    # File handler — rotates daily, keeps 7 days
    logger.add(
        "logs/app_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="DEBUG",
        rotation="00:00",
        retention="7 days",
        compression="zip",
    )

    logger.info(f"Logging initialized | level={settings.log_level}")
    return logger
