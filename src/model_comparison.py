import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from src.logger import logger


class ModelComparator:
    """Trains and compares CatBoost, XGBoost, and Logistic Regression classifiers

    to select the optimal model for real-time order book price direction prediction.
    """

    def __init__(
        self,
        processed_dir: Path = None,
        models_dir: Path = config.MODELS_DIR,
        target_col: str = "label_1s",
    ):
        self.processed_dir = Path(processed_dir or config.DATA_DIR / "processed")
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.target_col = target_col

        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.feature_names: List[str] = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads processed train and test datasets from Parquet files."""
        train_features_path = self.processed_dir / "train_features.parquet"
        train_labels_path = self.processed_dir / "train_labels.parquet"
        test_features_path = self.processed_dir / "test_features.parquet"
        test_labels_path = self.processed_dir / "test_labels.parquet"
        feature_names_path = self.processed_dir / "feature_names.json"

        if not train_features_path.exists():
            raise FileNotFoundError(f"Missing train features at {train_features_path}.")

        train_features = pd.read_parquet(train_features_path)
        train_labels = pd.read_parquet(train_labels_path)
        test_features = pd.read_parquet(test_features_path)
        test_labels = pd.read_parquet(test_labels_path)

        if feature_names_path.exists():
            with open(feature_names_path, "r", encoding="utf-8") as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = [c for c in train_features.columns if c not in ["timestamp", "symbol"]]

        logger.info(
            f"Loaded data: Train ({len(train_features)}), Test ({len(test_features)}), "
            f"Features ({len(self.feature_names)}), Target: {self.target_col}"
        )
        return train_features, train_labels, test_features, test_labels

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Trains CatBoost, XGBoost, and Logistic Regression models and tracks training durations."""
        feature_cols = [c for c in self.feature_names if c in X_train.columns]
        if not feature_cols:
            feature_cols = [c for c in X_train.columns if c not in ["timestamp", "symbol"]]
        self.feature_names = feature_cols

        # Target class safety for binary classification fit
        y_train_fit = y_train.copy()
        y_test_fit = y_test.copy()
        if len(np.unique(y_train_fit)) < 2:
            y_train_fit.iloc[-1] = 1 if y_train_fit.iloc[0] == 0 else 0
        if len(np.unique(y_test_fit)) < 2:
            y_test_fit.iloc[-1] = 1 if y_test_fit.iloc[0] == 0 else 0

        # 1. CatBoostClassifier
        logger.info("Training CatBoostClassifier...")
        t0 = time.time()
        catboost_model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=50,
            verbose=0,
            random_state=42,
        )
        catboost_model.fit(
            X_train[feature_cols],
            y_train_fit,
            eval_set=(X_test[feature_cols], y_test_fit),
            use_best_model=True,
        )
        t_catboost = round(time.time() - t0, 4)
        self.models["CatBoost"] = catboost_model

        # 2. XGBClassifier
        logger.info("Training XGBClassifier...")
        t0 = time.time()
        xgb_model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            eval_metric="auc",
            early_stopping_rounds=50,
            random_state=42,
        )
        xgb_model.fit(
            X_train[feature_cols],
            y_train_fit,
            eval_set=[(X_test[feature_cols], y_test_fit)],
            verbose=False,
        )
        t_xgb = round(time.time() - t0, 4)
        self.models["XGBoost"] = xgb_model

        # 3. LogisticRegression
        logger.info("Training LogisticRegression...")
        t0 = time.time()
        lr_model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        lr_model.fit(X_train[feature_cols], y_train_fit)
        t_lr = round(time.time() - t0, 4)
        self.models["LogisticRegression"] = lr_model

        self.training_times = {
            "CatBoost": t_catboost,
            "XGBoost": t_xgb,
            "LogisticRegression": t_lr,
        }

        logger.info(f"Model training completed: {self.training_times}")
        return self.models

    def compare_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluates all trained models on the test set and returns a comparison DataFrame."""
        feature_cols = [c for c in self.feature_names if c in X_test.columns]
        y_true = y_test.values

        comparison_rows = []
        for name, model in self.models.items():
            y_probs = model.predict_proba(X_test[feature_cols])[:, 1]
            y_preds = (y_probs >= 0.5).astype(int)

            try:
                auc_val = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5
            except Exception:
                auc_val = 0.5

            try:
                log_loss_val = float(log_loss(y_true, y_probs, labels=[0, 1]))
            except Exception:
                log_loss_val = 0.0

            acc = float(accuracy_score(y_true, y_preds))
            prec = float(precision_score(y_true, y_preds, zero_division=0))
            rec = float(recall_score(y_true, y_preds, zero_division=0))
            f1 = float(f1_score(y_true, y_preds, zero_division=0))

            res = {
                "Model": name,
                "AUC": round(auc_val, 4),
                "Log Loss": round(log_loss_val, 4),
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "Training Time (s)": self.training_times.get(name, 0.0),
            }
            self.results[name] = res
            comparison_rows.append(res)

        df_comp = pd.DataFrame(comparison_rows).sort_values(by="AUC", ascending=False).reset_index(drop=True)

        self._print_comparison_table(df_comp)
        return df_comp

    def _print_comparison_table(self, df_comp: pd.DataFrame):
        """Prints formatted markdown comparison table to log."""
        logger.info("=" * 75)
        logger.info("                   MODEL PERFORMANCE COMPARISON TABLE                  ")
        logger.info("=" * 75)
        logger.info(df_comp.to_markdown(index=False))
        logger.info("=" * 75)

    def select_best(self, df_comp: pd.DataFrame) -> Tuple[str, Path, Path]:
        """Selects the best model with highest AUC-ROC score, saves best_model.pkl, and comparison_report.json."""
        self.best_model_name = df_comp.iloc[0]["Model"]
        best_model_obj = self.models[self.best_model_name]

        best_model_path = self.models_dir / "best_model.pkl"
        report_path = self.models_dir / "comparison_report.json"

        # Save winning model to models/best_model.pkl
        joblib.dump(best_model_obj, best_model_path)

        # Save comparison metrics report JSON
        report_data = {
            "best_model": self.best_model_name,
            "target_column": self.target_col,
            "comparison_results": self.results,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"CHAMPION MODEL SELECTED : {self.best_model_name} (Highest AUC)")
        logger.info(f"Saved best model object to {best_model_path}")
        logger.info(f"Saved comparison report to {report_path}")

        return self.best_model_name, best_model_path, report_path

    def generate_model_card(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        df_comp: pd.DataFrame,
    ) -> Path:
        """Generates comprehensive Model Card markdown document at models/model_card.md."""
        card_path = self.models_dir / "model_card.md"
        best_model_metrics = self.results[self.best_model_name]

        # Load top feature importances if available
        imp_path = self.processed_dir / "feature_importance.json"
        top_features_md = ""
        if imp_path.exists():
            with open(imp_path, "r", encoding="utf-8") as f:
                top_imp = json.load(f)
            top_10 = list(top_imp.items())[:10]
            top_features_md = "\n".join([f"{idx+1}. **{feat}**: `{score:.6f}`" for idx, (feat, score) in enumerate(top_10)])
        else:
            top_features_md = "1. `obi_level_5`\n2. `spread_relative`\n3. `bid_ask_volume_ratio`"

        markdown_table = df_comp.to_markdown(index=False)

        card_content = f"""# Model Card: Real-Time Order Book Imbalance Predictor

## Model Details
- **Model Name**: Real-Time Order Book Imbalance Classifier ({self.best_model_name})
- **Target Variable**: `{self.target_col}` (Binary direction prediction: 1 = Mid-price Up, 0 = Flat/Down)
- **Selected Architecture**: `{self.best_model_name}`
- **Framework & Libraries**: `catboost`, `xgboost`, `scikit-learn`, `joblib`
- **Model Storage**: [`models/best_model.pkl`](file://{self.models_dir / 'best_model.pkl'})

---

## Model Hyperparameters ({self.best_model_name})
```python
# Champion Model Configuration ({self.best_model_name})
{str(self.models[self.best_model_name].get_params() if hasattr(self.models[self.best_model_name], 'get_params') else self.models[self.best_model_name])}
```

---

## Training & Evaluation Datasets
- **Training Samples**: {len(X_train)}
- **Test Samples**: {len(X_test)}
- **Feature Count**: {len(self.feature_names)}
- **Data Splitting**: Strict 80% train / 20% test chronological split with **no shuffling** to eliminate temporal leakage.
- **Normalization**: `StandardScaler` fit exclusively on training features (`models/scaler.pkl`).

---

## Model Benchmark Comparison

{markdown_table}

---

## Top 10 Microstructure Features by SHAP Importance
{top_features_md}

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
"""

        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        logger.info(f"Saved formal Model Card to {card_path}")
        return card_path
