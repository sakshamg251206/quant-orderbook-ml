# Real-Time Order Book Imbalance Predictor

A Python 3.10+ async data collection engine for real-time crypto order book imbalance prediction using Binance WebSockets.

## Project Structure

```
.
├── config.py             # Configuration loader (API keys, symbol, depth)
├── .env.example          # Environment variable template
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── data/                 # Directory for order book snapshot storage & logs
├── models/               # Directory for trained ML model artifacts
├── notebooks/            # Directory for Jupyter notebooks
└── src/                  # Source code
    ├── __init__.py
    ├── logger.py         # Structured logging configuration
    ├── storage.py        # Snapshot persistence (SQLite or Parquet)
    ├── collector.py      # Async Binance order book stream listener
    └── main.py           # Application entry point
```

## Setup & Installation

1. **Clone & Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and configure your parameters:
   ```bash
   cp .env.example .env
   ```

## Running the Data Collector

Start streaming order book depth and calculating imbalance metrics:
```bash
python -m src.main
```
