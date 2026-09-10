# Jupyter Notebooks Directory

Use this directory for exploratory data analysis, feature engineering, and model training for order book imbalance prediction.

Suggested workflow:
1. `01_eda_orderbook_imbalance.ipynb`: Load Parquet / SQLite snapshots from `../data` and analyze order book imbalance distribution.
2. `02_feature_engineering.ipynb`: Compute multi-level depth imbalances, spread velocity, and price trend indicators.
3. `03_model_training.ipynb`: Train ML models (e.g. XGBoost, LightGBM, LSTM) to predict short-term price movements from imbalance features.
