import sys
from pathlib import Path

import config
from src.logger import logger
from src.model_comparison import ModelComparator


def main():
    logger.info("Starting Multi-Model Comparison & Selection Pipeline...")
    target_column = "label_1s"

    comparator = ModelComparator(
        processed_dir=config.DATA_DIR / "processed",
        models_dir=config.MODELS_DIR,
        target_col=target_column,
    )

    try:
        # 1. Load processed train/test datasets
        X_train, y_train_df, X_test, y_test_df = comparator.load_data()

        # 2. Train CatBoost, XGBoost, and Logistic Regression models
        comparator.train_all_models(
            X_train=X_train,
            y_train=y_train_df[target_column],
            X_test=X_test,
            y_test=y_test_df[target_column],
        )

        # 3. Benchmark model performance on test set
        df_comp = comparator.compare_performance(
            X_test=X_test,
            y_test=y_test_df[target_column],
        )

        # 4. Select best model based on AUC-ROC & save models/best_model.pkl
        best_name, model_path, report_path = comparator.select_best(df_comp)

        # 5. Generate formal Model Card document (models/model_card.md)
        card_path = comparator.generate_model_card(
            X_train=X_train,
            X_test=X_test,
            df_comp=df_comp,
        )

        logger.info("=" * 70)
        logger.info("          MODEL COMPARISON & SELECTION FINISHED               ")
        logger.info("=" * 70)
        logger.info(f"Target Horizon       : {target_column}")
        logger.info(f"Champion Model Chosen: {best_name} (Highest AUC)")
        logger.info(f"Saved Best Model     : {model_path}")
        logger.info(f"Comparison Report JSON: {report_path}")
        logger.info(f"Formal Model Card    : {card_path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Model comparison pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
