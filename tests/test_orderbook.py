import json

import pytest

from orderbook_ml.collector import BookSynchronizer
from orderbook_ml.orderbook import (
    ApplyResult,
    DepthEvent,
    LocalOrderBook,
    OrderBookSyncError,
    build_snapshot_record,
)


def event(first: int, final: int, bids=(), asks=()) -> DepthEvent:
    return DepthEvent(first, final, [list(b) for b in bids], [list(a) for a in asks])


@pytest.fixture
def book() -> LocalOrderBook:
    b = LocalOrderBook("btcusdt")
    b.load_snapshot(100, [["10.0", "1"], ["9.9", "2"]], [["10.1", "1.5"], ["10.2", "3"]])
    return b


def test_symbol_is_upper_cased(book):
    assert book.symbol == "BTCUSDT"


def test_stale_events_are_ignored(book):
    assert book.apply(event(90, 100, bids=[("10.0", "99")])) is ApplyResult.STALE
    assert book.top_levels(1)[0] == [(10.0, 1.0)]


def test_first_event_may_straddle_the_snapshot_id(book):
    assert book.apply(event(95, 105, bids=[("10.05", "4")])) is ApplyResult.APPLIED
    assert book.last_update_id == 105
    assert book.top_levels(1)[0] == [(10.05, 4.0)]


def test_gap_is_detected_and_book_untouched(book):
    assert book.apply(event(102, 110, bids=[("10.05", "4")])) is ApplyResult.GAP
    assert book.last_update_id == 100
    assert book.top_levels(1)[0] == [(10.0, 1.0)]


def test_zero_quantity_removes_level(book):
    book.apply(event(101, 101, asks=[("10.1", "0")]))
    assert book.top_levels(5)[1] == [(10.2, 3.0)]


def test_top_levels_are_sorted_best_first(book):
    book.apply(event(101, 102, bids=[("9.5", "1")], asks=[("10.15", "1")]))
    bids, asks = book.top_levels(3)
    assert [p for p, _ in bids] == [10.0, 9.9, 9.5]
    assert [p for p, _ in asks] == [10.1, 10.15, 10.2]


def test_apply_before_snapshot_raises():
    with pytest.raises(OrderBookSyncError):
        LocalOrderBook("X").apply(event(1, 2))


def test_snapshot_record_matches_book(book):
    rec = book.snapshot_record(depth=2)
    assert rec["best_bid"] == 10.0 and rec["best_ask"] == 10.1
    assert rec["spread"] == pytest.approx(0.1)
    assert rec["total_bid_volume"] == 3.0 and rec["total_ask_volume"] == 4.5
    assert json.loads(rec["bids_json"]) == [[10.0, 1.0], [9.9, 2.0]]


def test_snapshot_record_requires_both_sides():
    assert build_snapshot_record("X", [], [(1.0, 1.0)]) is None


def test_depth_event_parsing_handles_combined_stream_payloads():
    msg = {"stream": "x", "data": {"e": "depthUpdate", "U": 5, "u": 7, "b": [], "a": []}}
    parsed = DepthEvent.from_message(msg)
    assert parsed is not None and (parsed.first_update_id, parsed.final_update_id) == (5, 7)
    assert DepthEvent.from_message({"result": None, "id": 1}) is None


class TestBookSynchronizer:
    def snapshot(self, last_id: int) -> dict:
        return {"lastUpdateId": last_id, "bids": [["10", "1"]], "asks": [["11", "1"]]}

    def test_buffers_until_snapshot_then_replays(self):
        sync = BookSynchronizer(LocalOrderBook("X"))
        sync.on_event(event(95, 99))  # older than snapshot: dropped on replay
        sync.on_event(event(100, 102, bids=[("10.5", "2")]))
        assert not sync.synced
        assert sync.snapshot_is_usable(100)
        sync.on_snapshot(self.snapshot(100))
        assert sync.synced and sync.book.last_update_id == 102
        assert sync.book.top_levels(1)[0] == [(10.5, 2.0)]
        sync.on_event(event(103, 104, asks=[("10.8", "1")]))
        assert sync.book.top_levels(1)[1] == [(10.8, 1.0)]

    def test_snapshot_older_than_buffer_is_rejected(self):
        sync = BookSynchronizer(LocalOrderBook("X"))
        sync.on_event(event(200, 201))
        assert not sync.snapshot_is_usable(150)

    def test_gap_after_sync_raises(self):
        sync = BookSynchronizer(LocalOrderBook("X"))
        sync.on_event(event(100, 101))
        sync.on_snapshot(self.snapshot(100))
        with pytest.raises(OrderBookSyncError):
            sync.on_event(event(110, 111))

    def test_buffer_that_does_not_connect_raises_and_resets(self):
        sync = BookSynchronizer(LocalOrderBook("X"))
        sync.on_event(event(150, 151))
        with pytest.raises(OrderBookSyncError):
            sync.on_snapshot(self.snapshot(100))
        assert not sync.synced and sync.book.last_update_id is None
