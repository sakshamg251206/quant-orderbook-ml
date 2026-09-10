# Model Card: Real-Time Order Book Imbalance Predictor

## Model Details
- **Model Name**: Real-Time Order Book Imbalance Classifier (CatBoost)
- **Target Variable**: `label_1s` (Binary direction prediction: 1 = Mid-price Up, 0 = Flat/Down)
- **Selected Architecture**: `CatBoost`
- **Framework & Libraries**: `catboost`, `xgboost`, `scikit-learn`, `joblib`
- **Model Storage**: [`models/best_model.pkl`](file://models/best_model.pkl)

---

## Model Hyperparameters (CatBoost)
```python
# Champion Model Configuration (CatBoost)
{'iterations': 500, 'learning_rate': 0.03, 'depth': 6, 'loss_function': 'Logloss', 'verbose': 0, 'eval_metric': 'AUC', 'random_state': 42, 'early_stopping_rounds': 50}
```

---

## Training & Evaluation Datasets
- **Training Samples**: 160
- **Test Samples**: 41
- **Feature Count**: 34
- **Data Splitting**: Strict 80% train / 20% test chronological split with **no shuffling** to eliminate temporal leakage.
- **Normalization**: `StandardScaler` fit exclusively on training features (`models/scaler.pkl`).

---

## Model Benchmark Comparison

| Model              |   AUC |   Log Loss |   Accuracy |   Precision |   Recall |   F1 Score |   Training Time (s) |
|:-------------------|------:|-----------:|-----------:|------------:|---------:|-----------:|--------------------:|
| CatBoost           |   0.5 |     0.282  |          1 |           0 |        0 |          0 |              0.1588 |
| XGBoost            |   0.5 |     0.0063 |          1 |           0 |        0 |          0 |              0.0458 |
| LogisticRegression |   0.5 |     0.0189 |          1 |           0 |        0 |          0 |              0.0248 |

---

## Top 10 Microstructure Features by SHAP Importance
1. **bid_volume_skew**: `0.161002`
2. **ask_volume_skew**: `0.114609`
3. **total_bid_volume**: `0.069581`
4. **ask_volume_decay**: `0.057508`
5. **obi_momentum_100ms**: `0.048224`
6. **bid_ask_volume_ratio**: `0.044193`
7. **weighted_mid_price_5**: `0.033472`
8. **micro_price**: `0.026812`
9. **micro_price_deviation**: `0.025642`
10. **obi_momentum_500ms**: `0.025431`

---

## Intended Use & Market Conditions
- **Intended Use**: High-frequency order book microstructure direction forecasting for execution algorithms and automated crypto market making.
- **Symbol Scope**: BTC/USDT spot order book (L2 100ms updates).
- **Latency Requirement**: Real-time sub-10ms feature extraction and inference loop.

---

## Limitations & Risks
1. **Regime Change Sensitivity**: Model is trained on specific market volatility regimes. Sudden liquidity shocks or market halt events may degrade performance.
2. **Execution Slippage**: Forecasts predict mid-price movements, not executable order fill prices across depth levels.
3. **Target Imbalance**: Low volatility periods can create single-class label concentration.
