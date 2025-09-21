Milestone: M0 — Kickoff & Environment

Decisions:
- ML framework: PyTorch
- Web framework: FastAPI

Commands run:
- venv create/activate, pip install -r requirements.txt
- (Optional) pip uninstall -y tensorflow
- (Optional) CUDA PyTorch install via cu124 index
- python scripts/sanity_check.py

Artifacts produced:
- Repo skeleton, .venv, requirements.txt (PyTorch+FastAPI), .env, sanity_check.py

Tests & results:
- sanity_check.py shows ML_FRAMEWORK=torch, WEB_FRAMEWORK=fastapi
- torch and fastapi import OK; python_binance import OK
- (Optional) CUDA available: True/False (not required)

Open issues / next actions:
- None for M0 once green
- Proceed to M1: Data Fetch & Normalization


Milestone: M1 — Data Fetch & Normalization (REST)

Decisions:
- None (using Binance REST, default features + target=close)

Commands run:
- python data_fetch.py --symbol BTCUSDT --interval 1h --limit 3000 --save-csv data\raw\BTCUSDT_1h.csv
- python data_fetch.py --symbol BTCUSDT --interval 1h --limit 3000 --normalize --scaler-out artifacts\scalers\BTCUSDT_1h_scaler.json

Artifacts produced:
- data/raw/BTCUSDT_1h.csv (optional)
- artifacts/scalers/BTCUSDT_1h_scaler.json

Tests & results:
- Shapes printed, normalized min/max within [0,1]
- Head/tail preview OK

Open issues / next actions:
- Proceed to M2: Windowing & Dataset
