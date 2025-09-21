# Real-time Crypto Candlestick + Forecast

Phases:
- Phase 1: Python LSTM (or PyTorch) forecasting on Binance OHLCV
- Phase 2: Frontend Chart.js + chartjs-chart-financial
- Phase 3: Integration via Flask/FastAPI REST + WebSocket

## Quick start (Windows)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python .\scripts\sanity_check.py
