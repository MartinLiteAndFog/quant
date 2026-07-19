from __future__ import annotations

import io
import importlib.util
import zipfile
from pathlib import Path

import pandas as pd


def _load_subject():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_binance_vision_ohlcv.py"
    spec = importlib.util.spec_from_file_location("download_binance_vision_ohlcv", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _archive(timestamp: int) -> bytes:
    row = [timestamp, 100, 102, 99, 101, 12, timestamp + 59_999, 0, 1, 0, 0, 0]
    csv = ",".join(str(value) for value in row) + "\n"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("BTCUSDT-1m.csv", csv)
    return output.getvalue()


def test_normalize_archive_supports_millisecond_and_microsecond_timestamps():
    expected = pd.Timestamp("2025-01-01T00:00:00Z")
    for timestamp in (int(expected.timestamp() * 1_000), int(expected.timestamp() * 1_000_000)):
        frame = subject.normalize_archive(_archive(timestamp))
        assert list(frame.columns) == ["ts", "open", "high", "low", "close", "volume"]
        assert frame["ts"].iloc[0] == expected
        assert frame["close"].iloc[0] == 101.0


def test_acquisition_plan_uses_public_monthly_urls_and_canonical_paths(tmp_path):
    plan = subject.acquisition_plan(
        ["BTC-USDT"],
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-02-28T23:59:00Z"),
        "1m",
        "https://data.binance.vision",
        tmp_path,
    )
    assert plan[0]["urls"] == [
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01.zip",
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2025-02.zip",
    ]
    assert "exchange=binance-vision/symbol=BTC-USDT/timeframe=1m" in str(plan[0]["output"])
