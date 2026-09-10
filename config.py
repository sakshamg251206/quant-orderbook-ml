import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Market & Order Book Configuration
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
ORDER_BOOK_DEPTH = int(os.getenv("ORDER_BOOK_DEPTH", "20"))

# Data Storage Configuration
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_TYPE = os.getenv("STORAGE_TYPE", "parquet").lower()
SNAPSHOT_INTERVAL_SEC = float(os.getenv("SNAPSHOT_INTERVAL_SEC", "0.1"))  # 100ms
PARQUET_BATCH_SIZE = int(os.getenv("PARQUET_BATCH_SIZE", "100"))  # Save Parquet file every 100 snapshots

SNAPSHOTS_DIR = DATA_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paths
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOKS_DIR = Path("./notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
