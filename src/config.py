from dataclasses import dataclass
import os
from dotenv import load_dotenv, find_dotenv

# Load variables from .env at project root (if present)
load_dotenv(find_dotenv())

@dataclass
class AppConfig:
    ml_framework: str = os.getenv("ML_FRAMEWORK", "tf")      # tf | torch
    web_framework: str = os.getenv("WEB_FRAMEWORK", "flask") # flask | fastapi
    symbol: str = os.getenv("SYMBOL", "BTCUSDT")
    interval: str = os.getenv("INTERVAL", "1h")
    lookback: int = int(os.getenv("LOOKBACK", "100"))
    horizon: int = int(os.getenv("HORIZON", "12"))
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))

CONFIG = AppConfig()
