import argparse
import asyncio
import sys
from pathlib import Path

import config
from src.logger import logger
from src.realtime_inference import RealTimePredictor


async def main_async(max_snapshots: int = None):
    logger.info("Initializing Real-Time Order Book Inference Pipeline...")

    predictor = RealTimePredictor(
        model_path=config.MODELS_DIR / "best_model.pkl",
        scaler_path=config.MODELS_DIR / "scaler.pkl",
        feature_names_path=config.DATA_DIR / "processed" / "feature_names.json",
        output_file=config.DATA_DIR / "predictions.parquet",
        buy_threshold=0.65,
        sell_threshold=0.35,
    )

    logger.info("Starting live 100ms inference stream loop...")
    await predictor.run(max_snapshots=max_snapshots)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Order Book Imbalance Inference System")
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Maximum number of 100ms snapshots to process before exiting (default: run indefinitely)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main_async(max_snapshots=args.max_snapshots))
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down real-time inference loop.")
    except Exception as e:
        logger.error(f"Real-time inference pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
