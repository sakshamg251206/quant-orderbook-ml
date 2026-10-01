"""Live Binance L2 order book stream.

Public market data needs no API key. The collector:

* opens the ``<symbol>@depth@100ms`` diff-depth websocket and buffers events,
* fetches a REST depth snapshot while buffering, then replays the buffer on top of it,
* keeps the local book in sync, rebuilding it from scratch whenever a sequence gap is detected,
* reconnects with capped exponential backoff on network errors,
* samples the book on a fixed clock (default every 100 ms) and yields snapshot rows.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import pandas as pd

from orderbook_ml.config import Settings
from orderbook_ml.logging_utils import get_logger
from orderbook_ml.orderbook import (
    ApplyResult,
    DepthEvent,
    LocalOrderBook,
    OrderBookSyncError,
)

logger = get_logger(__name__)

REST_SNAPSHOT_LIMIT = 1000
MAX_BUFFERED_EVENTS = 2000
MAX_BACKOFF_SEC = 30.0


class BookSynchronizer:
    """Turns a stream of depth events plus one REST snapshot into a synced local book.

    Pure state machine (no I/O) so the buffering rules can be tested in isolation.
    """

    def __init__(self, book: LocalOrderBook) -> None:
        self.book = book
        self.buffer: list[DepthEvent] = []
        self.synced = False

    def reset(self) -> None:
        self.book.reset()
        self.buffer.clear()
        self.synced = False

    def on_event(self, event: DepthEvent) -> None:
        """Buffers events until a snapshot is loaded, then applies them in order."""
        if not self.synced:
            if len(self.buffer) >= MAX_BUFFERED_EVENTS:
                raise OrderBookSyncError("Event buffer overflow while waiting for REST snapshot")
            self.buffer.append(event)
            return
        if self.book.apply(event) is ApplyResult.GAP:
            raise OrderBookSyncError(
                f"Sequence gap: expected first id <= {self.book.last_update_id + 1}"  # type: ignore[operator]
                f", got {event.first_update_id}"
            )

    def snapshot_is_usable(self, last_update_id: int) -> bool:
        """A snapshot is usable once it is not older than the first buffered event."""
        return bool(self.buffer) and last_update_id >= self.buffer[0].first_update_id - 1

    def on_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Seeds the book from a REST snapshot and replays the buffered events."""
        self.book.load_snapshot(snapshot["lastUpdateId"], snapshot["bids"], snapshot["asks"])
        pending, self.buffer = self.buffer, []
        applied_any = False
        for event in pending:
            result = self.book.apply(event)
            if result is ApplyResult.GAP:
                self.reset()
                raise OrderBookSyncError("Buffered events do not connect to the REST snapshot")
            applied_any = applied_any or result is ApplyResult.APPLIED
        self.synced = True
        logger.info(
            "Order book synchronised at update id %s (%d buffered events, applied=%s)",
            self.book.last_update_id,
            len(pending),
            applied_any,
        )


class BinanceOrderBookStream:
    """Async generator of order book snapshots sampled from the live Binance stream."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.symbol = settings.symbol
        self.book = LocalOrderBook(self.symbol)
        self.sync = BookSynchronizer(self.book)
        self.reconnects = 0

    @property
    def ws_url(self) -> str:
        return f"{self.settings.binance_ws_url}/ws/{self.symbol.lower()}@depth@100ms"

    async def _fetch_snapshot(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        url = f"{self.settings.binance_rest_url}/api/v3/depth"
        params = {"symbol": self.symbol, "limit": str(REST_SNAPSHOT_LIMIT)}
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                raise aiohttp.ClientResponseError(
                    resp.request_info, resp.history, status=resp.status, message=body
                )
            data: dict[str, Any] = await resp.json()
            return data

    async def _consume(self, session: aiohttp.ClientSession) -> None:
        """Runs one websocket connection until it closes or the book loses sync."""
        self.sync.reset()
        snapshot_task: asyncio.Task[dict[str, Any]] | None = None
        try:
            async with session.ws_connect(self.ws_url, heartbeat=30) as ws:
                logger.info("Connected to %s", self.ws_url)
                async for msg in ws:
                    if msg.type is aiohttp.WSMsgType.ERROR:
                        raise ConnectionError(f"Websocket error: {ws.exception()}")
                    if msg.type is not aiohttp.WSMsgType.TEXT:
                        continue
                    event = DepthEvent.from_message(json.loads(msg.data))
                    if event is None:
                        continue
                    self.sync.on_event(event)

                    if self.sync.synced:
                        continue
                    if snapshot_task is None:
                        snapshot_task = asyncio.create_task(self._fetch_snapshot(session))
                    elif snapshot_task.done():
                        snapshot = snapshot_task.result()
                        snapshot_task = None
                        if self.sync.snapshot_is_usable(int(snapshot["lastUpdateId"])):
                            self.sync.on_snapshot(snapshot)
                        else:
                            logger.debug("REST snapshot older than buffered events; refetching")
            raise ConnectionError("Websocket closed by server")
        finally:
            if snapshot_task is not None and not snapshot_task.done():
                snapshot_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await snapshot_task

    async def _maintain(self) -> None:
        """Keeps the local book alive forever, reconnecting with exponential backoff."""
        backoff = 1.0
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    await self._consume(session)
                except asyncio.CancelledError:
                    raise
                except aiohttp.ClientResponseError as exc:
                    if exc.status in (400, 401, 403, 451):
                        hint = (
                            " Binance restricts some regions (HTTP 451); set "
                            "OBML_BINANCE_REST_URL / OBML_BINANCE_WS_URL to a reachable endpoint."
                            if exc.status in (403, 451)
                            else ""
                        )
                        raise RuntimeError(
                            f"Binance rejected the request (HTTP {exc.status}): "
                            f"{exc.message}.{hint}"
                        ) from exc
                    logger.warning("REST error %s; reconnecting in %.0fs", exc, backoff)
                except (OrderBookSyncError, aiohttp.ClientError, ConnectionError, OSError) as exc:
                    logger.warning("%s; resynchronising in %.0fs", exc, backoff)
                was_synced = self.sync.synced
                self.sync.reset()
                self.reconnects += 1
                await asyncio.sleep(backoff)
                backoff = 1.0 if was_synced else min(MAX_BACKOFF_SEC, backoff * 2)

    async def snapshots(self, max_snapshots: int | None = None) -> AsyncIterator[dict]:
        """Yields one snapshot row every ``snapshot_interval_ms`` while the book is synced."""
        loop = asyncio.get_running_loop()
        interval = self.settings.snapshot_interval_sec
        maintainer = asyncio.create_task(self._maintain(), name="orderbook-maintainer")
        emitted = 0
        try:
            next_tick = loop.time()
            while max_snapshots is None or emitted < max_snapshots:
                if maintainer.done():
                    maintainer.result()  # re-raises fatal errors
                    return
                if self.sync.synced:
                    record = self.book.snapshot_record(
                        self.settings.depth, timestamp=pd.Timestamp.now(tz="UTC")
                    )
                    if record is not None:
                        emitted += 1
                        yield record
                next_tick += interval
                delay = next_tick - loop.time()
                if delay < 0:  # fell behind (e.g. slow consumer): skip missed ticks
                    next_tick = loop.time()
                    delay = 0
                await asyncio.sleep(delay)
        finally:
            maintainer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await maintainer
