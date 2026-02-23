import logging
import sys

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
