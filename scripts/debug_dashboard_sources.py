"""Diagnostic: fetch /api/dashboard/chart from deployed dashboard, write _debug to log."""
from __future__ import annotations
import json, sys, time
from urllib.request import urlopen

LOG_PATH = "/Users/martinpeter/Desktop/quant/.cursor/debug-2cd5ae.log"
SESSION = "2cd5ae"

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else input("Railway dashboard base URL (e.g. https://quant-production-xxxx.up.railway.app): ").strip().rstrip("/")
    url = f"{base}/api/dashboard/chart?hours=336&max_points=5000"
    print(f"Fetching {url} ...")
    resp = urlopen(url, timeout=30)
    payload = json.loads(resp.read().decode())

    debug = payload.get("_debug", {})
    markers = payload.get("markers", [])
    levels = payload.get("levels", {})

    entries = []

    def log(hyp, msg, data):
        entries.append(json.dumps({
            "sessionId": SESSION,
            "id": f"log_{int(time.time()*1000)}_{hyp}",
            "timestamp": int(time.time() * 1000),
            "location": "debug_dashboard_sources.py",
            "message": msg,
            "data": data,
            "runId": "run1",
            "hypothesisId": hyp,
        }, default=str))

    log("H1", "fills_and_trades_counts", {
        "markers_from_trades_parquet": debug.get("markers_from_trades_parquet"),
        "markers_from_live_fills": debug.get("markers_from_live_fills"),
        "markers_total": debug.get("markers_total_after_merge"),
    })
    log("H2", "renko_bar_range", {
        "renko_bars_count": debug.get("renko_bars_count"),
        "oldest_bar_ts": debug.get("oldest_bar_ts"),
        "newest_bar_ts": debug.get("newest_bar_ts"),
    })
    log("H3", "marker_newest_timestamps", {
        "newest_5_marker_times": debug.get("marker_newest_5_times"),
        "newest_bar_ts": debug.get("newest_bar_ts"),
    })
    log("H4", "levels_and_expected_entry", {
        "levels_keys": debug.get("levels_keys"),
        "levels_side": debug.get("levels_side"),
        "levels_entry_bar_ts": debug.get("levels_entry_bar_ts"),
        "levels_entry_px": debug.get("levels_entry_px"),
        "levels_ts": debug.get("levels_ts"),
        "expected_entry": debug.get("expected_entry"),
        "live_entry_marker": debug.get("live_entry_marker"),
        "open_position": debug.get("open_position"),
    })
    log("H5", "diary_and_equity", {
        "diary_count": debug.get("diary_count"),
        "diary_source": debug.get("diary_source"),
        "equity_count": debug.get("equity_count"),
    })

    # Dump all marker texts for inspection
    marker_texts = [{"time": m.get("time"), "text": m.get("text"), "shape": m.get("shape")} for m in markers[-20:]]
    log("ALL", "last_20_markers", {"markers": marker_texts})

    with open(LOG_PATH, "a") as f:
        for e in entries:
            f.write(e + "\n")

    print(f"\nWrote {len(entries)} log entries to {LOG_PATH}")
    print(f"\n=== _debug summary ===")
    print(json.dumps(debug, indent=2, default=str))

if __name__ == "__main__":
    main()
