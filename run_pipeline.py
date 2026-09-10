import sys
from pathlib import Path

import config
from src.logger import logger
from src.data_pipeline import DataPipeline


def main():
    logger.info("Starting End-to-End Data Pipeline...")
    pipeline = DataPipeline(
        data_dir=config.SNAPSHOTS_DIR,
        processed_dir=config.DATA_DIR / "processed",
        models_dir=config.MODELS_DIR,
        train_ratio=0.8,
    )

    try:
        summary = pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("             END-TO-END PIPELINE EXPORT SUMMARY                ")
    logger.info("=" * 70)
    logger.info(f"Raw Snapshots Processed   : {summary['total_raw_snapshots']:,}")
    logger.info(f"Data Quality Score        : {summary['quality_score']:.2f}%")
    logger.info(f"Train Set Samples (80%)   : {summary['train_samples']:,}")
    logger.info(f"Test Set Samples (20%)    : {summary['test_samples']:,}")
    logger.info(f"Microstructure Features   : {summary['num_features']}")
    logger.info(f"Target Label Columns      : {', '.join(summary['label_columns'])}")
    logger.info("-" * 70)
    logger.info("Exported Dataset Files:")
    for k, path in summary["exported_files"].items():
        logger.info(f"  • {k:<16s}: {path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
