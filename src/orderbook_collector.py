import asyncio
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from binance import AsyncClient, BinanceSocketManager

import config
from src.logger import logger


class AsyncBinanceOrderBook:
    """Async L2 Order Book manager that reconstructs order book state from Binance WebSocket

    depth update streams and saves 100ms snapshots to Parquet files.
    """

    # PyArrow explicit schema for Parquet storage
    PARQUET_SCHEMA = pa.schema([
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("symbol", pa.string()),
        ("bids_json", pa.string()),
        ("asks_json", pa.string()),
        ("best_bid", pa.float64()),
        ("best_ask", pa.float64()),
        ("spread", pa.float64()),
        ("total_bid_volume", pa.float64()),
        ("total_ask_volume", pa.float64()),
    ])

    def __init__(
        self,
        symbol: str = config.SYMBOL,
        depth: int = config.ORDER_BOOK_DEPTH,
        snapshots_dir: Path = config.SNAPSHOTS_DIR,
        snapshot_interval: float = config.SNAPSHOT_INTERVAL_SEC,
        batch_size: int = config.PARQUET_BATCH_SIZE,
    ):
        self.symbol = symbol.upper()
        self.depth = depth
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_interval = snapshot_interval
        self.batch_size = batch_size

        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None

        # Local L2 order book dictionaries: price (float) -> qty (float)
        self._bids: Dict[float, float] = {}
        self._asks: Dict[float, float] = {}
        self.last_update_id: int = 0
        self.is_synced: bool = False
        self.is_running: bool = False
        self.is_fetching_snapshot: bool = False

        # Buffers & tasks
        self.snapshot_buffer: List[dict] = []
        self.snapshot_count: int = 0
        self._snapshot_task: Optional[asyncio.Task] = None

    async def _init_client(self):
        """Initializes the Binance AsyncClient."""
        self.client = await AsyncClient.create(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_API_SECRET,
        )
        self.bsm = BinanceSocketManager(self.client)

    async def _fetch_snapshot(self):
        """Fetches the L2 order book REST snapshot from Binance to seed order book state."""
        if self.is_fetching_snapshot:
            return
        self.is_fetching_snapshot = True
        try:
            logger.info(f"Fetching REST L2 order book snapshot for {self.symbol}...")
            snapshot = await self.client.get_order_book(symbol=self.symbol, limit=1000)

            self.last_update_id = int(snapshot["lastUpdateId"])
            self._bids = {float(price): float(qty) for price, qty in snapshot["bids"]}
            self._asks = {float(price): float(qty) for price, qty in snapshot["asks"]}
            self.is_synced = False

            logger.info(
                f"Fetched REST snapshot. LastUpdateId: {self.last_update_id} | "
                f"Bids: {len(self._bids)} levels, Asks: {len(self._asks)} levels"
            )
        except Exception as e:
            logger.error(f"Failed to fetch REST order book snapshot: {e}", exc_info=True)
        finally:
            self.is_fetching_snapshot = False

    def _apply_depth_diff(self, bids: List[Tuple[str, str]], asks: List[Tuple[str, str]]):
        """Applies price level depth updates to local bids and asks."""
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self._bids.pop(price, None)
            else:
                self._bids[price] = qty

        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self._asks.pop(price, None)
            else:
                self._asks[price] = qty

    def _process_message(self, msg: dict):
        """Updates local order book state from incremental WebSocket depthUpdate messages.

        Follows official Binance L2 order book synchronization rules.
        """
        data = msg.get("data", msg)
        event_type = data.get("e")

        if event_type != "depthUpdate":
            return

        first_update_id = int(data.get("U", 0))
        final_update_id = int(data.get("u", 0))
        bids = data.get("b", [])
        asks = data.get("a", [])

        if self.last_update_id == 0:
            # REST snapshot not loaded yet
            return

        # Binance Sync Rule 1: Drop updates older than lastUpdateId
        if final_update_id <= self.last_update_id:
            return

        if not self.is_synced:
            # The first processed event should have U <= lastUpdateId+1 AND u >= lastUpdateId+1
            if first_update_id <= self.last_update_id + 1 and final_update_id >= self.last_update_id + 1:
                self._apply_depth_diff(bids, asks)
                self.last_update_id = final_update_id
                self.is_synced = True
                logger.info(f"Order book synchronized with WebSocket stream at update ID {self.last_update_id}")
            elif first_update_id > self.last_update_id + 1:
                logger.warning(
                    f"Gap before initial sync (FirstUpdateID: {first_update_id} > LastUpdateID+1: {self.last_update_id + 1}). "
                    "Re-fetching snapshot..."
                )
                asyncio.create_task(self._fetch_snapshot())
            return

        # Binance Sync Rule 2: Verify continuity of update ID sequence
        if first_update_id > self.last_update_id + 1:
            logger.warning(
                f"Order book update gap detected! FirstUpdateID: {first_update_id}, "
                f"Expected <= {self.last_update_id + 1}. Resynchronizing..."
            )
            self.is_synced = False
            asyncio.create_task(self._fetch_snapshot())
            return

        # Apply depth updates and advance state update ID
        self._apply_depth_diff(bids, asks)
        self.last_update_id = final_update_id

    def _get_current_snapshot_record(self) -> Optional[dict]:
        """Generates a snapshot record matching the target Parquet schema."""
        if not self.is_synced or not self._bids or not self._asks:
            return None

        # Sort top N bids (descending) and top N asks (ascending)
        sorted_bids = sorted(
            [(p, q) for p, q in self._bids.items() if q > 0],
            key=lambda x: x[0],
            reverse=True,
        )[: self.depth]

        sorted_asks = sorted(
            [(p, q) for p, q in self._asks.items() if q > 0],
            key=lambda x: x[0],
        )[: self.depth]

        if not sorted_bids or not sorted_asks:
            return None

        best_bid = float(sorted_bids[0][0])
        best_ask = float(sorted_asks[0][0])
        spread = float(best_ask - best_bid)

        total_bid_volume = float(sum(q for _, q in sorted_bids))
        total_ask_volume = float(sum(q for _, q in sorted_asks))

        now_utc = pd.Timestamp.now(tz="UTC")

        return {
            "timestamp": now_utc,
            "symbol": self.symbol,
            "bids_json": json.dumps([[p, q] for p, q in sorted_bids]),
            "asks_json": json.dumps([[p, q] for p, q in sorted_asks]),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
        }

    def _save_snapshot(self):
        """Takes a snapshot of current order book state and appends to batch buffer."""
        record = self._get_current_snapshot_record()
        if record is None:
            return

        self.snapshot_buffer.append(record)
        self.snapshot_count += 1

        # Requirement: Log every 1000 snapshots
        if self.snapshot_count % 1000 == 0:
            logger.info(
                f"[SNAPSHOT #{self.snapshot_count}] Symbol: {self.symbol} | "
                f"Best Bid: {record['best_bid']:.2f} | Best Ask: {record['best_ask']:.2f} | "
                f"Spread: {record['spread']:.2f} | Bid Vol: {record['total_bid_volume']:.4f} | "
                f"Ask Vol: {record['total_ask_volume']:.4f}"
            )

        if len(self.snapshot_buffer) >= self.batch_size:
            self._flush_snapshots_to_parquet()

    def _flush_snapshots_to_parquet(self):
        """Writes buffered snapshots to a Parquet file."""
        if not self.snapshot_buffer:
            return

        try:
            df = pd.DataFrame(self.snapshot_buffer)
            table = pa.Table.from_pandas(df, schema=self.PARQUET_SCHEMA)

            timestamp_str = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{self.symbol}_ob_{timestamp_str}.parquet"
            file_path = self.snapshots_dir / file_name

            pq.write_table(table, file_path, compression="SNAPPY")
            logger.info(f"Flushed {len(self.snapshot_buffer)} snapshots to Parquet file: {file_path.name}")
            self.snapshot_buffer.clear()
        except Exception as e:
            logger.error(f"Failed to write snapshots to Parquet: {e}", exc_info=True)

    async def _snapshot_loop(self):
        """Background loop taking snapshots every 100ms (snapshot_interval)."""
        logger.info(f"Started 100ms snapshot loop (interval: {self.snapshot_interval}s)")
        try:
            while self.is_running:
                start_time = asyncio.get_event_loop().time()
                self._save_snapshot()
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0.0, self.snapshot_interval - elapsed)
                await asyncio.sleep(sleep_time)
        except asyncio.CancelledError:
            logger.info("Snapshot loop task cancelled.")
        except Exception as e:
            logger.error(f"Error in snapshot loop: {e}", exc_info=True)

    async def run(self):
        """Main execution loop: connects to WebSocket stream, handles reconnections, and processes messages."""
        self.is_running = True
        reconnect_delay = 1.0

        while self.is_running:
            try:
                if self.client is None:
                    await self._init_client()

                logger.info(f"Connecting to Binance WebSocket L2 depth stream ({self.symbol.lower()}@depth@100ms)...")
                stream_name = f"{self.symbol.lower()}@depth@100ms"
                depth_socket = self.bsm.multiplex_socket([stream_name])

                # Start REST snapshot fetch
                await self._fetch_snapshot()

                # Start snapshot generator loop task
                if self._snapshot_task is None or self._snapshot_task.done():
                    self._snapshot_task = asyncio.create_task(self._snapshot_loop())

                async with depth_socket as stream:
                    logger.info("Connected to WebSocket stream. Processing depth updates...")
                    reconnect_delay = 1.0  # Reset backoff delay

                    while self.is_running:
                        msg = await stream.recv()
                        if not msg:
                            continue
                        self._process_message(msg)

            except asyncio.CancelledError:
                logger.info("Order book collector main loop cancelled.")
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}. Reconnecting in {reconnect_delay}s...", exc_info=True)
                self.is_synced = False
                logger.info("Reconnecting to WebSocket depth stream...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(30.0, reconnect_delay * 2.0)  # Exponential backoff
            finally:
                if self.client:
                    try:
                        await self.client.close_connection()
                    except Exception:
                        pass
                    self.client = None

        # Clean up remaining snapshots on exit
        if self._snapshot_task and not self._snapshot_task.done():
            self._snapshot_task.cancel()
        self._flush_snapshots_to_parquet()
        logger.info("Order book collector shut down completely.")

    def stop(self):
        """Stops the collector and snapshot loop."""
        self.is_running = False

    async def stream_snapshots(self):
        """Async generator yielding 100ms snapshot records from live WebSocket stream."""
        self.is_running = True
        snapshot_task = asyncio.create_task(self.run())
        try:
            while self.is_running:
                rec = self._get_current_snapshot_record()
                if rec is not None:
                    yield rec
                await asyncio.sleep(self.snapshot_interval)
        finally:
            self.stop()
            if not snapshot_task.done():
                snapshot_task.cancel()

    async def close(self):
        """Closes collector resources."""
        self.stop()
        if self.client:
            try:
                await self.client.close_connection()
            except Exception:
                pass
            self.client = None

