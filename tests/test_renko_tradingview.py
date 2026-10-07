import pandas as pd

from quant.features.renko import renko_from_close


def _df(closes):
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=len(closes), freq="5min", tz="UTC"),
            "close": closes,
        }
    )


def test_traditional_reversal_requires_two_boxes():
    bricks = renko_from_close(_df([100.0, 100.1, 100.2, 100.1]), box=0.1)

    # A one-box pullback from 100.2 to 100.1 must not create a reversal brick.
    assert bricks["dir"].tolist() == [1, 1]
    assert bricks["close"].tolist() == [100.1, 100.2]


def test_traditional_reversal_uses_tradingview_gap_geometry():
    bricks = renko_from_close(_df([100.0, 100.1, 100.2, 100.0]), box=0.1)
    last = bricks.iloc[-1]

    # TradingView Traditional Renko: after an up brick closing at 100.2,
    # a down reversal opens one box lower and closes two boxes lower.
    assert int(last["dir"]) == -1
    assert float(last["open"]) == 100.1
    assert float(last["close"]) == 100.0


def test_multiple_bricks_from_one_source_bar_get_unique_millisecond_timestamps():
    df = pd.DataFrame(
        {
            "ts": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T00:05:00Z"),
            ],
            "close": [100.0, 100.4],
        }
    )
    bricks = renko_from_close(df, box=0.1)

    assert len(bricks) == 4
    assert bricks["ts"].is_unique
    assert bricks["ts"].tolist() == [
        pd.Timestamp("2026-01-01T00:05:00Z"),
        pd.Timestamp("2026-01-01T00:05:00.001Z"),
        pd.Timestamp("2026-01-01T00:05:00.002Z"),
        pd.Timestamp("2026-01-01T00:05:00.003Z"),
    ]


def test_observed_tradingview_reversal_example():
    closes = [
        97.2,
        97.3,
        97.4,
        97.5,
        97.6,
        97.7,
        97.8,
        97.9,
        98.0,
        98.1,
        98.2,
        98.3,
        98.1,
    ]
    bricks = renko_from_close(_df(closes), box=0.1)
    last = bricks.iloc[-1]

    # Observed in the fresh TradingView Kraken SOLUSD.PM Renko export:
    # previous close 98.3 -> first down reversal brick 98.2 -> 98.1.
    assert int(last["dir"]) == -1
    assert float(last["open"]) == 98.2
    assert float(last["close"]) == 98.1
