#!/usr/bin/env bash
set -e

# Persistency: if a GCS bucket is mounted at /data, mirror folders there
mkdir -p /data/reminders /data/parsed_data /data/embeddings || true
[ -e reminders ]    || ln -s /data/reminders reminders
[ -e parsed_data ]  || ln -s /data/parsed_data parsed_data
[ -e embeddings ]   || ln -s /data/embeddings embeddings

# Minimal Streamlit secrets (UI auth). Add Drive later if needed.
mkdir -p ~/.streamlit
cat > ~/.streamlit/secrets.toml <<EOF
app_user = "${APP_USER:-admin123}"
app_pass = "${APP_PASS:-BestOrg123@#}"
EOF

# Start Streamlit UI (uses chat_ceo.py)
exec python -m streamlit run chat_ceo.py --server.port="${PORT:-8080}" --server.address=0.0.0.0
