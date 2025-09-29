FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends \
      tini bash ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code from repo root (Cloud Build runs in repo root)
# Adjust paths only if your files live elsewhere.
COPY ../../chat_ceo.py ./
COPY ../../answer_with_rag.py ../../semantic_search.py ../../embed_and_store.py ./
COPY ../../file_parser.py ../../chunk_utils.py ../../reminders_extractor.py ./
COPY ../../knowledge_curator.py ../../gdrive_uploader.py ./

COPY entrypoint-ui.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/entrypoint.sh"]
