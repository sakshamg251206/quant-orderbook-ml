"""Runs the real collector against a local fake of the Binance REST + websocket API."""

import asyncio
import json

import pytest
from aiohttp import web

from orderbook_ml.collector import BinanceOrderBookStream
from orderbook_ml.config import Settings


class FakeBinance:
    """Serves a depth snapshot and a stream of contiguous diff events.

    On the first connection the stream skips an update id after a few events, which the
    collector must detect as a gap and recover from by reconnecting.
    """

    def __init__(self) -> None:
        self.next_id = 100
        self.connections = 0
        self.snapshot_requests = 0

    async def depth(self, request: web.Request) -> web.Response:
        self.snapshot_requests += 1
        assert request.query["symbol"] == "BTCUSDT"
        return web.json_response(
            {
                "lastUpdateId": self.next_id,
                "bids": [["100.0", "1.0"], ["99.0", "2.0"]],
                "asks": [["101.0", "1.0"], ["102.0", "2.0"]],
            }
        )

    async def stream(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.connections += 1
        first_connection = self.connections == 1
        for i in range(200):
            if ws.closed:
                break
            if first_connection and i == 30:
                self.next_id += 5  # missed updates -> sequence gap
            start = self.next_id + 1
            self.next_id += 2
            qty = f"{1 + i % 5}.0"
            event = {
                "e": "depthUpdate",
                "U": start,
                "u": self.next_id,
                "b": [["100.0", qty]],
                "a": [["101.0", "1.5"]],
            }
            await ws.send_str(json.dumps(event))
            await asyncio.sleep(0.005)
        await ws.close()
        return ws


@pytest.fixture
async def fake_binance():
    fake = FakeBinance()
    app = web.Application()
    app.router.add_get("/api/v3/depth", fake.depth)
    app.router.add_get("/ws/btcusdt@depth@100ms", fake.stream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    yield fake, port
    await runner.cleanup()


async def test_collector_syncs_and_recovers_from_gaps(fake_binance, monkeypatch):
    monkeypatch.setattr("orderbook_ml.collector.MAX_BACKOFF_SEC", 0.1)
    fake, port = fake_binance
    settings = Settings(
        symbol="BTCUSDT",
        snapshot_interval_ms=20,
        binance_rest_url=f"http://127.0.0.1:{port}",
        binance_ws_url=f"ws://127.0.0.1:{port}",
    )
    stream = BinanceOrderBookStream(settings)
    records = []
    async with asyncio.timeout(20):
        async for record in stream.snapshots(max_snapshots=120):
            records.append(record)

    assert len(records) == 120
    assert stream.reconnects >= 1 and fake.snapshot_requests >= 2
    assert all(r["best_bid"] == 100.0 and r["best_ask"] == 101.0 for r in records)
    assert all(r["spread"] == 1.0 for r in records)
    assert {json.loads(r["asks_json"])[0][1] for r in records} == {1.5}
    ids = [r["last_update_id"] for r in records]
    assert ids == sorted(ids)


def test_plain_http_only_allowed_for_localhost():
    from orderbook_ml.config import ConfigError

    Settings(binance_rest_url="http://localhost:9000", binance_ws_url="ws://127.0.0.1:9000")
    with pytest.raises(ConfigError):
        Settings(binance_rest_url="http://api.example.com")
