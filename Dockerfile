# Lightweight Python base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs (faiss-cpu needs BLAS; PyPDF etc. may need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Cloud Run provides $PORT; Streamlit must bind to 0.0.0.0
ENV PORT=8080 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8080

# IMPORTANT: start Streamlit, not Uvicorn
# Use bash -lc so $PORT expands into the arg value
CMD ["bash","-lc","streamlit run chat_ceo.py --server.port $PORT --server.address 0.0.0.0"]

