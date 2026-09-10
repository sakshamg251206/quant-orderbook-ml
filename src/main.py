import asyncio
import signal
import sys
import config
from src.logger import logger
from src.storage import DataStorage
from src.collector import OrderBookCollector


async def main():
    logger.info("=" * 60)
    logger.info("Starting Binance Real-time Order Book Imbalance Predictor")
    logger.info(f"Symbol: {config.SYMBOL} | Depth: {config.ORDER_BOOK_DEPTH} | Storage: {config.STORAGE_TYPE}")
    logger.info("=" * 60)

    storage = DataStorage()
    collector = OrderBookCollector(storage=storage)

    # Handle graceful shutdown signals
    loop = asyncio.get_running_loop()

    def shutdown():
        logger.info("Shutdown signal received. Stopping collector...")
        collector.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            # Signal handlers not implemented on some platforms (e.g. Windows)
            pass

    try:
        await collector.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Application exited successfully.")


if __name__ == "__main__":
    asyncio.run(main())
