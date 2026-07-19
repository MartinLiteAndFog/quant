"""Parity tests for quant.backtest.fast_ct against the recorded reference run.

The reference artifacts live under data/runs/ (gitignored), so these tests
skip when they are absent instead of failing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.backtest.fast_ct import fast_flip_trades, fast_imba_signals

_REPO = Path(__file__).resolve().parent.parent
_SWEEP = _REPO / "data" / "runs" / "2026-07-04_SOLUSDT_1y_top3_renko_box_sweep"
_REF_RUN = _REPO / "data" / "runs" / "2026-07-04_1y_box0p08_lb250_ct_ttp1"

pytestmark = pytest.mark.skipif(
    not (_SWEEP / "renko" / "box_0p08.parquet").exists() or not (_REF_RUN / "trades.parquet").exists(),
    reason="reference run artifacts not present (data/ is gitignored)",
)


def _reference_bars_and_impulses():
    """Reproduce the reference run's exact bar preparation and signal alignment.

    The reference engine (flip_engine._ensure_cols) deduplicates colliding brick
    timestamps with an UNSTABLE sort + keep=last, and aligns saved signals by
    exact timestamp match, silently dropping ns-offset signals. fast_ct itself
    uses the clean per-brick convention; this fixture only exists to prove
    engine-level parity on identical inputs.
    """
    from quant.strategies.flip_engine import _ensure_cols

    bricks = pd.read_parquet(_SWEEP / "renko" / "box_0p08.parquet")
    bars = _ensure_cols(bricks, ["ts", "close"], name="bars")

    sig = pd.read_json(_SWEEP / "signals" / "box_0p08_lb250.jsonl", lines=True)
    sig["ts"] = pd.to_datetime(sig["ts"], utc=True, format="ISO8601")
    sig_map = dict(zip(sig["ts"], sig["signal"]))
    impulse = pd.DatetimeIndex(bars["ts"]).map(lambda t: sig_map.get(t, 0)).values.astype(np.int64)
    return bars, impulse


def test_flip_engine_parity_with_reference_run():
    bars, impulse = _reference_bars_and_impulses()
    c = bars["close"].values.astype(float)
    h = bars["high"].values.astype(float)
    l = bars["low"].values.astype(float)

    res = fast_flip_trades(
        c, h, l, impulse,
        fee_bps=12.0, ttp_trail_pct=0.01, min_sl_pct=0.0095, max_sl_pct=0.03, swing_lookback=50,
    )
    ref = pd.read_parquet(_REF_RUN / "trades.parquet")

    assert len(res["pnl_pct"]) == len(ref) == 1273
    assert np.allclose(res["pnl_pct"], ref["pnl_pct"].values, atol=1e-12)
    assert np.allclose(res["entry_px"], ref["entry_px"].values, atol=1e-9)
    assert np.allclose(res["exit_px"], ref["exit_px"].values, atol=1e-9)

    eq = np.cumprod(1.0 + res["pnl_pct"])
    assert (eq[-1] - 1.0) * 100.0 == pytest.approx(278.0996, abs=1e-3)


def test_imba_signals_parity_with_compute_imba_signals():
    from quant.strategies.imba import ImbaParams, compute_imba_signals

    bricks = pd.read_parquet(_SWEEP / "renko" / "box_0p08.parquet")
    o = bricks["open"].values.astype(float)
    c = bricks["close"].values.astype(float)
    h = np.maximum(o, c)
    l = np.minimum(o, c)

    idx, val = fast_imba_signals(h, l, c, 250)

    # Reference implementation needs unique timestamps (ns offsets) to keep all bricks.
    ohlc = pd.DataFrame({"ts": bricks["ts"].values, "open": o, "high": h, "low": l, "close": c})
    grp = ohlc["ts"].astype("int64")
    ohlc["ts"] = ohlc["ts"] + pd.to_timedelta(ohlc.groupby(grp).cumcount(), unit="ns")
    ref = compute_imba_signals(ohlc, ImbaParams(lookback=250))

    assert len(idx) == len(ref)
    assert np.array_equal(val, ref["signal"].values)
    # same bricks: the ns-adjusted timestamps at our indices must equal the reference ts
    assert np.array_equal(
        ohlc["ts"].iloc[idx].values,
        pd.to_datetime(ref["ts"], utc=True).values,
    )
