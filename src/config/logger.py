"""
This module sets up the application configuration

It utilizies Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from pathlib import Path

from loguru import logger
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class LoggerSettings(BaseSettings):
    """
    Logger Configuration settings for the application.

    Attributes:
        log_level: Logging level for the application
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    log_level: str


def configure_logging(log_level: str) -> None:
    """
    Configure the logging for the application.

    Args:
        log_level: the log level to be set for the logger.

    Returns:
        None
    """
    logger.remove()
    logger.add(
        "logs/app.log",
        rotation="1 day",
        retention="2 days",
        compression="zip",
        level=log_level,
    )


configure_logging(log_level=LoggerSettings().log_level)
