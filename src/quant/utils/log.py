import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict

def get_logger(name: str = "quant", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    try:
        from rich.logging import RichHandler
        handler = RichHandler(rich_tracebacks=True, markup=True)
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


@dataclass
class _ThrottleState:
    last_ts: float
    suppressed: int = 0


_THROTTLE_LOCK = threading.Lock()
_THROTTLE_MAP: Dict[str, _ThrottleState] = {}


def log_throttled(
    logger: logging.Logger,
    level: int,
    key: str,
    interval_sec: float,
    msg: str,
    *args,
) -> bool:
    """
    Log no more than once per interval for a given key.
    Returns True when message was emitted, False when suppressed.
    """
    interval = float(max(0.0, interval_sec))
    now = time.time()
    suppressed = 0
    emit = False

    with _THROTTLE_LOCK:
        st = _THROTTLE_MAP.get(key)
        if st is None or interval <= 0.0 or (now - st.last_ts) >= interval:
            emit = True
            if st is not None:
                suppressed = int(st.suppressed)
            _THROTTLE_MAP[key] = _ThrottleState(last_ts=now, suppressed=0)
        else:
            st.suppressed += 1

    if not emit:
        return False

    if suppressed > 0:
        logger.log(level, f"{msg} [suppressed {suppressed} similar messages]", *args)
    else:
        logger.log(level, msg, *args)
    return True
