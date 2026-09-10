import datetime
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import config
from src.logger import logger

Base = declarative_base()


class OrderBookSnapshot(Base):
    """SQLAlchemy Model for SQLite order book snapshots."""
    __tablename__ = "order_book_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    last_update_id = Column(Integer)
    bid_price_top = Column(Float)
    bid_qty_top = Column(Float)
    ask_price_top = Column(Float)
    ask_qty_top = Column(Float)
    mid_price = Column(Float)
    bid_ask_spread = Column(Float)
    total_bid_volume = Column(Float)
    total_ask_volume = Column(Float)
    order_book_imbalance = Column(Float)


class DataStorage:
    """Handles storage of order book snapshots to SQLite or Parquet files."""

    def __init__(self, storage_type: str = config.STORAGE_TYPE, data_dir: Path = config.DATA_DIR):
        self.storage_type = storage_type
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if self.storage_type == "sqlite":
            db_path = self.data_dir / "order_book.db"
            self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Initialized SQLite storage at {db_path}")
        else:
            self.parquet_dir = self.data_dir / "snapshots"
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initialized Parquet storage at {self.parquet_dir}")

    def save_snapshot(self, record: Dict[str, Any]) -> None:
        """Saves a single snapshot dictionary to storage."""
        try:
            if self.storage_type == "sqlite":
                session = self.Session()
                snapshot = OrderBookSnapshot(**record)
                session.add(snapshot)
                session.commit()
                session.close()
            else:
                df = pd.DataFrame([record])
                timestamp_str = record["timestamp"].strftime("%Y%m%d_%H%M%S_%f")
                file_path = self.parquet_dir / f"{record['symbol']}_{timestamp_str}.parquet"
                df.to_parquet(file_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save snapshot to {self.storage_type}: {e}", exc_info=True)
