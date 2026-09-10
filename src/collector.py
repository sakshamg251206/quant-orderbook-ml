import asyncio
import datetime
from typing import Dict, Any
import numpy as np
from binance import AsyncClient, BinanceSocketManager

import config
from src.logger import logger
from src.storage import DataStorage


class OrderBookCollector:
    """Async collector for streaming Binance order book depth snapshots and calculating imbalance."""

    def __init__(
        self,
        symbol: str = config.SYMBOL,
        depth: int = config.ORDER_BOOK_DEPTH,
        storage: DataStorage = None,
    ):
        self.symbol = symbol.upper()
        self.depth = depth
        self.storage = storage or DataStorage()
        self.is_running = False

    def calculate_imbalance(self, bids: list, asks: list) -> Dict[str, Any]:
        """Calculates order book imbalance and key quantitative metrics.

        Imbalance Formula:
        I = (V_bid - V_ask) / (V_bid + V_ask)
        Range: [-1.0, 1.0] where 1.0 indicates strong buy pressure and -1.0 indicates strong sell pressure.
        """
        bid_prices = np.array([float(b[0]) for b in bids[: self.depth]])
        bid_qtys = np.array([float(b[1]) for b in bids[: self.depth]])

        ask_prices = np.array([float(a[0]) for a in asks[: self.depth]])
        ask_qtys = np.array([float(a[1]) for a in asks[: self.depth]])

        total_bid_vol = float(np.sum(bid_qtys))
        total_ask_vol = float(np.sum(ask_qtys))

        total_vol = total_bid_vol + total_ask_vol
        imbalance = (total_bid_vol - total_ask_vol) / total_vol if total_vol > 0 else 0.0

        bid_top = float(bid_prices[0]) if len(bid_prices) > 0 else 0.0
        bid_qty_top = float(bid_qtys[0]) if len(bid_qtys) > 0 else 0.0
        ask_top = float(ask_prices[0]) if len(ask_prices) > 0 else 0.0
        ask_qty_top = float(ask_qtys[0]) if len(ask_qtys) > 0 else 0.0

        mid_price = (bid_top + ask_top) / 2.0 if (bid_top > 0 and ask_top > 0) else 0.0
        spread = ask_top - bid_top

        return {
            "symbol": self.symbol,
            "timestamp": datetime.datetime.utcnow(),
            "last_update_id": 0,
            "bid_price_top": bid_top,
            "bid_qty_top": bid_qty_top,
            "ask_price_top": ask_top,
            "ask_qty_top": ask_qty_top,
            "mid_price": mid_price,
            "bid_ask_spread": spread,
            "total_bid_volume": total_bid_vol,
            "total_ask_volume": total_ask_vol,
            "order_book_imbalance": imbalance,
        }

    async def start(self):
        """Connects to Binance WebSocket stream and starts collecting depth snapshots."""
        self.is_running = True
        logger.info(f"Initializing Binance AsyncClient for symbol: {self.symbol}")

        client = await AsyncClient.create(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
        )
        bsm = BinanceSocketManager(client)

        # Connect to depth socket (e.g. depth20@100ms or standard depth socket)
        depth_socket = bsm.depth_socket(self.symbol, depth=str(self.depth))

        try:
            async with depth_socket as ds:
                logger.info(f"Connected to order book WebSocket stream for {self.symbol}")
                while self.is_running:
                    msg = await ds.recv()
                    if msg:
                        bids = msg.get("bids", [])
                        asks = msg.get("asks", [])
                        if bids and asks:
                            record = self.calculate_imbalance(bids, asks)
                            record["last_update_id"] = msg.get("lastUpdateId", 0)

                            self.storage.save_snapshot(record)
                            logger.debug(
                                f"[{self.symbol}] Imbalance: {record['order_book_imbalance']:.4f} | Mid: {record['mid_price']:.2f}"
                            )
        except asyncio.CancelledError:
            logger.info("Collector task cancelled.")
        except Exception as e:
            logger.error(f"Error in Binance WebSocket depth stream: {e}", exc_info=True)
        finally:
            await client.close_connection()
            logger.info("Closed Binance AsyncClient connection.")

    def stop(self):
        """Stops the collection loop."""
        self.is_running = False
