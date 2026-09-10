import asyncio
import signal
import sys

import config
from src.logger import logger
from src.orderbook_collector import AsyncBinanceOrderBook


async def main():
    logger.info("=" * 70)
    logger.info("Starting Binance L2 Order Book Data Collection System")
    logger.info(
        f"Symbol: {config.SYMBOL} | Depth: {config.ORDER_BOOK_DEPTH} | "
        f"Interval: {config.SNAPSHOT_INTERVAL_SEC * 1000:.0f}ms | Parquet Dir: {config.SNAPSHOTS_DIR}"
    )
    logger.info("=" * 70)

    collector = AsyncBinanceOrderBook(
        symbol=config.SYMBOL,
        depth=config.ORDER_BOOK_DEPTH,
        snapshots_dir=config.SNAPSHOTS_DIR,
        snapshot_interval=config.SNAPSHOT_INTERVAL_SEC,
        batch_size=config.PARQUET_BATCH_SIZE,
    )

    loop = asyncio.get_running_loop()

    def handle_shutdown():
        logger.info("Termination signal received. Shutting down Order Book Collector...")
        collector.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_shutdown)
        except NotImplementedError:
            pass  # Signal handlers not implemented on Windows OS

    try:
        await collector.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught in main runner.")
    finally:
        logger.info("Execution finished.")


if __name__ == "__main__":
    asyncio.run(main())
