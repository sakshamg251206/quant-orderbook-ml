import argparse
import subprocess
import sys
import time
from pathlib import Path

import config
from src.logger import logger


def run_cmd(command: str):
    """Executes a shell command cleanly."""
    logger.info(f"Running command: {command}")
    res = subprocess.run(command, shell=True)
    if res.returncode != 0:
        logger.error(f"Command failed with exit code {res.returncode}: {command}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Master Execution & Interview Demo Script")
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run end-to-end pipeline (collector -> features -> training -> comparison) before dashboard",
    )
    args = parser.parse_args()

    python_bin = sys.executable

    print("\n" + "=" * 70)
    print("🚀 QUANTITATIVE ORDER BOOK ML & REAL-TIME INFERENCE SYSTEM")
    print("=" * 70 + "\n")

    if args.full_pipeline:
        logger.info("Executing Full Quantitative Pipeline...")
        run_cmd(f"{python_bin} run_collector.py --max-snapshots 200")
        run_cmd(f"{python_bin} run_pipeline.py")
        run_cmd(f"{python_bin} run_training.py")
        run_cmd(f"{python_bin} run_comparison.py")
        logger.info("Pipeline Execution Complete!")

    logger.info("Starting Real-Time Dashboard...")
    print("\n💡 INTERVIEWER DEMO INSTRUCTIONS:")
    print("-----------------------------------------------------------------")
    print("1. Streamlit Dashboard will launch automatically in your browser.")
    print("2. Navigate through Metrics, Order Book Depth, OBI History & Signals.")
    print("3. Press Ctrl+C in terminal to stop.")
    print("-----------------------------------------------------------------\n")

    time.sleep(1)
    run_cmd("streamlit run app.py")


if __name__ == "__main__":
    main()
