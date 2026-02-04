"""
Configuration management using Pydantic settings

This module manages all configuration from environment variables and YAML files.
Settings are validated at startup to catch configuration errors early.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    These settings control the bot's behavior and can be customized
    without changing code. All settings have sensible defaults.
    """
    
    # Environment
    ENVIRONMENT: str = Field(default="development", description="development, staging, or production")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL connection string")
    
    # Polymarket API (optional for Phase 1)
    POLYMARKET_API_KEY: Optional[str] = Field(default=None, description="Polymarket API key")
    POLYMARKET_API_SECRET: Optional[str] = Field(default=None, description="Polymarket API secret")
    
    # Bot Configuration
    MODEL_SAVE_PATH: str = Field(default="./models", description="Directory to save model checkpoints")
    
    # Risk Management Settings (Phase 1 defaults - virtual money)
    INITIAL_CAPITAL: float = Field(default=1000.0, description="Starting capital for backtesting")
    MAX_DAILY_LOSS_USD: float = Field(default=20.0, description="Maximum daily loss in USD")
    MAX_DAILY_LOSS_PCT: float = Field(default=0.05, description="Maximum daily loss as percentage")
    MAX_WEEKLY_LOSS_USD: float = Field(default=50.0, description="Maximum weekly loss in USD")
    MAX_WEEKLY_LOSS_PCT: float = Field(default=0.15, description="Maximum weekly loss as percentage")
    MAX_TOTAL_DRAWDOWN: float = Field(default=0.30, description="Maximum total drawdown before pause")
    MAX_POSITION_SIZE_PCT: float = Field(default=0.20, description="Maximum position size as % of capital")
    MAX_OPEN_POSITIONS: int = Field(default=3, description="Maximum concurrent open positions")
    MIN_MARKET_LIQUIDITY: float = Field(default=500.0, description="Minimum market liquidity in USD")
    
    # Circuit Breaker Settings
    MAX_CONSECUTIVE_LOSSES: int = Field(default=3, description="Pause after N consecutive losses")
    MIN_WIN_RATE_THRESHOLD: float = Field(default=0.35, description="Minimum win rate over 20 trades")
    NO_TRADE_WINDOW_BEFORE_RESOLUTION: int = Field(default=7200, description="No trading N seconds before resolution")
    
    # Training Settings
    TRAINING_EPISODES: int = Field(default=100000, description="Default number of training episodes")
    CHECKPOINT_FREQUENCY: int = Field(default=10000, description="Save checkpoint every N episodes")
    EVAL_FREQUENCY: int = Field(default=10000, description="Evaluate every N episodes")
    
    # Transaction Costs
    TRANSACTION_COST_PCT: float = Field(default=0.001, description="Transaction cost percentage (0.1%)")
    MAX_SLIPPAGE_PCT: float = Field(default=0.02, description="Maximum acceptable slippage (2%)")
    
    # API Configuration (for monitoring API)
    API_PORT: int = Field(default=8000, description="API server port")
    CORS_ORIGINS: str = Field(default="http://localhost:3000", description="Allowed CORS origins")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of the allowed values"""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"ENVIRONMENT must be one of {allowed}")
        return v
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v_upper
    
    @field_validator("MAX_POSITION_SIZE_PCT", "MAX_TOTAL_DRAWDOWN", "TRANSACTION_COST_PCT")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        """Ensure percentage values are between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create model save directory if it doesn't exist
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)


# Singleton pattern for settings
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get application settings (singleton)
    
    This function ensures we only load settings once and reuse them
    throughout the application lifecycle.
    
    Returns:
        Settings: Application settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Force reload settings from environment
    
    Useful for testing or when configuration changes at runtime.
    
    Returns:
        Settings: Fresh settings instance
    """
    global _settings
    _settings = Settings()
    return _settings
