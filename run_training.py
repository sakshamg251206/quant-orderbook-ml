import sys
from pathlib import Path

import config
from src.logger import logger
from src.model_training import CatBoostTrainer


def main():
    logger.info("Initializing CatBoost Model Training Pipeline...")
    target_column = "label_1s"

    trainer = CatBoostTrainer(
        processed_dir=config.DATA_DIR / "processed",
        models_dir=config.MODELS_DIR,
        plots_dir=config.DATA_DIR.parent / "plots",
        target_col=target_column,
    )

    try:
        # 1. Load processed train/test datasets
        X_train, y_train_df, X_test, y_test_df = trainer.load_data()

        # 2. Train CatBoostClassifier
        trainer.train_catboost(
            X_train=X_train,
            y_train=y_train_df[target_column],
            X_test=X_test,
            y_test=y_test_df[target_column],
            iterations=500,
            depth=6,
            learning_rate=0.03,
            early_stopping_rounds=50,
            verbose=100,
        )

        # 3. Compute evaluation metrics
        metrics = trainer.evaluate(X_test, y_test_df[target_column])

        # 4. Generate evaluation plots
        trainer.plot_training_history()
        trainer.plot_confusion_matrix(X_test, y_test_df[target_column])
        trainer.plot_calibration_curve(X_test, y_test_df[target_column])

        # 5. Save model and metrics artifacts
        model_path, metrics_path = trainer.save_model(metrics)

        logger.info("=" * 65)
        logger.info("             CATBOOST TRAINING COMPLETED SUCCESSFULLY          ")
        logger.info("=" * 65)
        logger.info(f"Target Horizon       : {target_column}")
        logger.info(f"AUC-ROC Score        : {metrics['auc_roc']:.4f}")
        logger.info(f"Log Loss             : {metrics['log_loss']:.4f}")
        logger.info(f"Accuracy             : {metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Trained Model File   : {model_path}")
        logger.info(f"Training Metrics JSON: {metrics_path}")
        logger.info("Plots Exported       : plots/training_history.png, confusion_matrix.png, calibration_curve.png")
        logger.info("=" * 65)

    except Exception as e:
        logger.error(f"Model training pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
