from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    exchange: str = os.getenv("EXCHANGE", "kucoin")
    market_type: str = os.getenv("MARKET_TYPE", "spot")  # spot | swap (later)
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")
    timeframe: str = os.getenv("TIMEFRAME", "1m")
    limit: int = int(os.getenv("LIMIT", "500"))
    data_dir: str = os.getenv("DATA_DIR", "data/raw")

CFG = Config()
