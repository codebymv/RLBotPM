"""
Database connection and ORM models

Defines the database schema for storing:
- Training runs and episodes
- Trades and positions
- Market data
- Model checkpoints
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON, ForeignKey, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from typing import Optional
import os

from ..core.config import get_settings


Base = declarative_base()


class TrainingRun(Base):
    """
    Represents a complete training session
    
    A training run is created when you start training and tracks
    overall performance across all episodes.
    """
    __tablename__ = 'training_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    total_episodes = Column(Integer, default=0)
    best_sharpe_ratio = Column(Float, nullable=True)
    best_episode_reward = Column(Float, nullable=True)
    status = Column(String(50), default='running')  # running, completed, failed, interrupted
    config_snapshot = Column(JSON, nullable=True)  # Store hyperparameters used
    
    # Relationships
    episodes = relationship("Episode", back_populates="training_run", cascade="all, delete-orphan")
    models = relationship("ModelCheckpoint", back_populates="training_run", cascade="all, delete-orphan")


class Episode(Base):
    """
    Represents a single episode (one complete game)
    
    An episode is a sequence of steps where the agent trades
    in one or more markets until termination.
    """
    __tablename__ = 'episodes'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey('training_runs.id'), nullable=False)
    episode_num = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    
    # Performance metrics
    total_reward = Column(Float, nullable=True)
    total_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    num_trades = Column(Integer, default=0)
    num_winning_trades = Column(Integer, default=0)
    final_capital = Column(Float, nullable=True)
    
    # Episode metadata
    markets_traded = Column(JSON, nullable=True)  # List of market IDs
    extra_metadata = Column("metadata", JSON, nullable=True)  # Additional info
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="episodes")
    trades = relationship("Trade", back_populates="episode", cascade="all, delete-orphan")


class Trade(Base):
    """
    Represents a single trade action
    
    Records every buy/sell decision made by the agent
    for complete audit trail.
    """
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(Integer, ForeignKey('episodes.id'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Market information
    market_id = Column(String(255), nullable=False)
    market_question = Column(Text, nullable=True)
    
    # Trade details
    action = Column(Integer, nullable=False)  # Action space index
    action_name = Column(String(50), nullable=True)  # Human-readable action
    position_size = Column(Float, nullable=False)  # Amount traded
    price = Column(Float, nullable=False)  # Execution price
    side = Column(String(10), nullable=False)  # 'buy' or 'sell'
    
    # Outcomes
    immediate_reward = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)  # Profit/loss if position closed
    
    # Agent reasoning (for interpretability)
    confidence = Column(Float, nullable=True)  # Model confidence
    features_snapshot = Column(JSON, nullable=True)  # State features at time of trade
    
    # Relationships
    episode = relationship("Episode", back_populates="trades")


class Market(Base):
    """
    Stores legacy Polymarket market data (deprecated).
    """
    __tablename__ = 'markets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    polymarket_id = Column(String(255), unique=True, nullable=False)
    question = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False)
    resolution_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    
    # Market metadata
    outcome = Column(String(50), nullable=True)  # Final outcome (Yes/No/Invalid)
    volume = Column(Float, nullable=True)
    liquidity = Column(Float, nullable=True)
    
    # Historical data
    price_history = Column(JSON, nullable=True)  # Time series of prices
    extra_metadata = Column("metadata", JSON, nullable=True)
    
    # Data collection
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelCheckpoint(Base):
    """
    Tracks saved model checkpoints
    
    Records model snapshots during training for
    versioning and rollback capability.
    """
    __tablename__ = 'model_checkpoints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey('training_runs.id'), nullable=False)
    
    # Model information
    episode_num = Column(Integer, nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Performance at checkpoint
    sharpe_ratio = Column(Float, nullable=True)
    avg_reward = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    
    # Metadata
    is_best = Column(Boolean, default=False)  # Best model so far
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    training_run = relationship("TrainingRun", back_populates="models")


class CryptoSymbol(Base):
    """
    Stores crypto symbols available from exchanges.
    """
    __tablename__ = 'crypto_symbols'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)  # coinbase, kraken, etc.
    symbol = Column(String(50), nullable=False)  # e.g., BTC-USD or XBTUSD
    base_asset = Column(String(20), nullable=True)
    quote_asset = Column(String(20), nullable=True)
    status = Column(String(20), default='active')
    extra_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class CryptoCandle(Base):
    """
    Stores OHLCV candles for crypto symbols.
    """
    __tablename__ = 'crypto_candles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String(50), nullable=False)
    symbol = Column(String(50), nullable=False)
    interval = Column(String(10), nullable=False)  # 1m, 5m, 1h, 1d
    timestamp = Column(DateTime, nullable=False)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class KalshiMarketHistory(Base):
    """
    Stores Kalshi market price history for RL training.

    Schema aligned with KalshiTradingEnv: ticker, timestamp, yes_price,
    volume, outcome (1=YES, 0=NO, null if not settled), close_time for time_to_expiry.
    """
    __tablename__ = "kalshi_market_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    yes_price = Column(Float, nullable=False)  # 0-1 probability
    volume = Column(Float, default=0)
    open_interest = Column(Integer, default=0)
    outcome = Column(Integer, nullable=True)  # 1=YES, 0=NO, null if not settled
    close_time = Column(DateTime, nullable=True)  # market expiration for time_to_expiry
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# Database engine and session factory
def get_engine():
    """Create database engine from settings"""
    settings = get_settings()
    return create_engine(
        settings.DATABASE_URL,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,
        max_overflow=20
    )


def get_session_factory():
    """Create session factory"""
    engine = get_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """
    Initialize database schema
    
    Creates all tables if they don't exist.
    Safe to call multiple times.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    print("[OK] Database schema initialized")


def get_db_session():
    """
    Get a database session
    
    Usage:
        session = get_db_session()
        try:
            # Do database operations
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    """
    SessionFactory = get_session_factory()
    return SessionFactory()


# Context manager for database sessions
class DatabaseSession:
    """
    Context manager for database sessions
    
    Usage:
        with DatabaseSession() as session:
            # Do database operations
            # Automatically commits on success, rolls back on error
    """
    
    def __enter__(self):
        self.session = get_db_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
