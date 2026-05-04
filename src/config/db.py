"""
This module sets up the application configuration

It utilizies Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine


BASE_DIR = Path(__file__).resolve().parents[2]

class DbSettings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        model_config: Model config, loaded from .env file
        db_conn_str: Database connection string
        table_name: Name of the table
    """

    model_config = SettingsConfigDict(
        env_file= BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    db_conn_str: str
    table_name: str


db_settings = DbSettings()

engine = create_engine(db_settings.db_conn_str)