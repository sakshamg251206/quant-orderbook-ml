import sys
from pathlib import Path

import config
from src.logger import logger
from src.data_validator import DataValidator


def main():
    logger.info("Initializing Data Quality Validation...")
    validator = DataValidator(data_dir=config.SNAPSHOTS_DIR)

    # 1. Load Parquet Data
    df = validator.load_data()
    if df.empty:
        logger.error("No data loaded. Please run 'python3 run_collector.py' first to collect order book snapshots.")
        sys.exit(1)

    # 2. Check timestamp gaps
    gaps = validator.check_timestamp_gaps(df, threshold_ms=200.0)
    if not gaps.empty:
        logger.warning(f"Found {len(gaps)} timestamp gap(s) exceeding 200ms threshold.")

    # 3. Check spread integrity (crossed / negative spreads)
    integrity_issues = validator.check_spread_integrity(df)
    if not integrity_issues.empty:
        logger.warning(f"Found {len(integrity_issues)} spread integrity violation(s).")

    # 4. Check volume integrity
    volume_issues = validator.check_volume_integrity(df)
    if not volume_issues.empty:
        logger.warning(f"Found {len(volume_issues)} zero/invalid volume snapshot(s).")

    # 5. Check spread outliers
    outliers = validator.check_spread_outliers(df, multiplier=10.0)
    if not outliers.empty:
        logger.warning(f"Found {len(outliers)} spread outlier(s).")

    # 6. Generate summary report & score
    report = validator.generate_report(df)

    # 7. Create visualization plot
    chart_path = validator.create_visualization(df)
    logger.info(f"Data validation completed successfully. Quality score: {report['quality_score']:.2f}%. Report image: {chart_path}")


if __name__ == "__main__":
    main()
