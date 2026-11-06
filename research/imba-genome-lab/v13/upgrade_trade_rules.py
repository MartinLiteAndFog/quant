#!/usr/bin/env python3
"""Upgrade an existing self-contained Genome v13 HTML file with the new trading rules.

The large gzip-embedded backtest data stays untouched; only the source JS ledger is replaced.
This works on the exported standalone HTML without rebuilding the 7,326 backtests.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

APP_PATTERN = re.compile(
    r"/\* Quant Genome v13 — every candidate belongs to an individually simulated daily curve\."
    r".*?\bstartApp\(\);",
    re.DOTALL,
)

def upgrade(source: Path, target: Path, javascript: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Input HTML does not exist: {source}")
    if not javascript.is_file():
        raise FileNotFoundError(f"Patched dashboard JS does not exist: {javascript}")
    html = source.read_text(encoding="utf-8")
    replacement = javascript.read_text(encoding="utf-8")
    matches = list(APP_PATTERN.finditer(html))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected precisely one embedded Genome v13 app; found {len(matches)}. "
            "Check that the input is the correct v13 standalone HTML."
        )
    original_app = matches[0].group()
    if "SCREENSHOT_PRESETS" not in original_app or "window.__QUANT_V13_READY" not in original_app:
        raise RuntimeError("Embedded v13 app does not contain the expected snapshot and ready markers.")
    if "function runTradeSweep()" not in replacement or "function simulatePortfolio(" not in replacement:
        raise RuntimeError("JS source is missing the new trading-rule implementation.")
    if source.resolve() == target.resolve():
        raise RuntimeError("Choose a separate output file; original HTML must remain untouched.")
    html = html[:matches[0].start()] + replacement + html[matches[0].end():]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    print(f"Generated {target} ({target.stat().st_size:,} bytes)")
    print("Preserved original embedded 2022–2026 data; swapped only the application logic.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path, help="Original exported Genome v13 HTML")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML; default: *_topn_sweep.html")
    args = parser.parse_args()
    target = args.out or args.html.with_name(args.html.stem + "_topn_sweep.html")
    upgrade(args.html, target, Path(__file__).with_name("app_v13.js"))
