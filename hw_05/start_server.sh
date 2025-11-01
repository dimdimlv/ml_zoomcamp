#!/bin/bash
cd /Users/dimdim/PycharmProjects/ml_zoomcamp/hw_05
uv run python -m uvicorn service:app --host 127.0.0.1 --port 8000
