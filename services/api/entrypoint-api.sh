#!/usr/bin/env bash
set -e

# Persistency mount (same scheme used by app.py bootstrap)
mkdir -p /data/reminders /data/parsed_data /data/embeddings || true
[ -e reminders ]    || ln -s /data/reminders reminders
[ -e parsed_data ]  || ln -s /data/parsed_data parsed_data
[ -e embeddings ]   || ln -s /data/embeddings embeddings

# Start FastAPI (app.py exposes /health, /ask, /reminder, /refresh)
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8080}"
