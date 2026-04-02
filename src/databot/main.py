"""DATABOT entrypoint — runs Renko pipeline on a polling loop + health server."""
from __future__ import annotations

import logging
import os
import sys
import threading
import time

import uvicorn

from databot.config import DatabotConfig
from databot.renko_pipeline import refresh_renko

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("databot")


def _run_pipeline_loop(cfg: DatabotConfig) -> None:
    """Polling loop that refreshes Renko for every configured symbol."""
    log.info(
        "pipeline loop starting: symbols=%s box=%s days_back=%d poll_sec=%s",
        cfg.symbols, cfg.renko_box, cfg.renko_days_back, cfg.poll_sec,
    )

    while True:
        for symbol in cfg.symbols:
            try:
                result = refresh_renko(symbol, cfg)
                if result.get("ok"):
                    log.info(
                        "renko refresh ok: symbol=%s bricks=%d last_close=%s redis=%s pg=%s",
                        symbol,
                        result.get("bricks", 0),
                        result.get("last_close"),
                        (result.get("redis") or {}).get("ok"),
                        (result.get("postgres") or {}).get("ok"),
                    )
                else:
                    log.warning("renko refresh failed: symbol=%s reason=%s", symbol, result.get("reason"))
            except Exception as exc:
                log.error("renko refresh exception: symbol=%s err=%s", symbol, exc, exc_info=True)

        time.sleep(max(5.0, cfg.poll_sec))


def _run_health_server(cfg: DatabotConfig) -> None:
    """Run the FastAPI health server in a thread."""
    uvicorn.run(
        "databot.health:app",
        host="0.0.0.0",
        port=cfg.health_port,
        log_level="warning",
    )


def main() -> None:
    cfg = DatabotConfig()

    health_thread = threading.Thread(
        target=_run_health_server, args=(cfg,), daemon=True, name="databot-health",
    )
    health_thread.start()
    log.info("health server started on port %d", cfg.health_port)

    _run_pipeline_loop(cfg)


if __name__ == "__main__":
    main()
