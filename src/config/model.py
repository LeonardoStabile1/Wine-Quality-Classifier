"""
This module sets up the application configuration

It utilizies Pydantic's BaseSettings for configuration management,
allowing settings to be read from environment variables and a .env file.
"""

from pathlib import Path

from pydantic import DirectoryPath
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class ModelSettings(BaseSettings):
    """
    Configuration settings for the application.

    Attributes:
        model_config: Model config, loaded from .env file
        model_path: Filesystem path to the model
        model_name: Name of the ML model
        report_path: Filesystem path to reports
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",)
    )

    model_path: DirectoryPath
    model_name: str
    report_path: DirectoryPath


model_settings = ModelSettings()
