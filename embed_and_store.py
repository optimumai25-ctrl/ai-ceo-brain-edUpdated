# embed_and_store.py
# Build a local vector index (no FAISS). Saves:
#  - embeddings/vectors.npy     (float32 [N, D])
#  - embeddings/meta.jsonl      (one JSON per vector)
#  - embeddings/embedding_report.csv

from __future__ import annotations
import os, re, json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# OpenAI embeddings (new SDK preferred, fallback legacy)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536 dims
PARSED_DIR = Path("parsed_data")
EMB_DIR = Path("embeddings")
EMB_DIR.mkdir(exist_ok=True, parents=True)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))      # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) # chars

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out, i, n = [], 0, len(text)
    while i < n:
        out.append(text[i:i+size])
        if i + size >= n:
            break
        i += max(1, size - overlap)
    return out

def _file_date_from_name(name: str) -> str | None:
    m = DATE_RE.search(name)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None

def _collect_inputs() -> Tuple[List[str], List[Dict]]:
    texts: List[str] = []
    metas: List[Dict] = []
    if not PARSED_DIR.exists():
        return texts, metas

    for path in PARSED_DIR.rglob("*.txt"):
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        chunks = _chunk(raw)
        folder = path.parent.name
        fdate = _file_date_from_name(path.name)
        for cid, ch in enumerate(chunks):
            metas.append({
                "filename": path.name,
                "path": str(path),
                "folder": folder,                         # e.g., "meetings", "reminders", "finance"
                "chunk_id": cid,
                "file_date": fdate,                       # "YYYY-MM-DD" or None
                "chars": len(ch),
                "text_preview": ch[:500]
            })
            texts.append(ch)
    return texts, metas

def _embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vecs: List[List[float]] = []
    B = 128
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        if _use_client:
            resp = _client.embeddings.create(model=EMBED_MODEL, input=batch)  # type: ignore
            vecs.extend([d.embedding for d in resp.data])
        else:
            resp = openai.Embedding.create(model=EMBED_MODEL, input=batch)  # type: ignore
            vecs.extend([d["embedding"] for d in resp["data"]])
    arr = np.asarray(vecs, dtype=np.float32)
    return arr

def _write_report(metas: List[Dict]):
    if not metas:
        return
    df = pd.DataFrame(metas)
    g = df.groupby("filename").agg(chunks=("chunk_id", "count"), chars=("chars", "sum")).reset_index()
    g.to_csv(EMB_DIR / "embedding_report.csv", index=False)

def main():
    texts, metas = _collect_inputs()
    if not texts:
        # clear stale index if any
        (EMB_DIR / "vectors.npy").unlink(missing_ok=True)
        (EMB_DIR / "meta.jsonl").unlink(missing_ok=True)
        (EMB_DIR / "embedding_report.csv").unlink(missing_ok=True)
        print("No parsed_data/*.txt found. Index cleared.")
        return

    vectors = _embed(texts)
    if vectors.shape[0] != len(metas):
        raise RuntimeError("Embedding count mismatch.")

    # Persist
    np.save(EMB_DIR / "vectors.npy", vectors)
    with (EMB_DIR / "meta.jsonl").open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    _write_report(metas)
    print(f"Wrote {vectors.shape[0]} embeddings to {EMB_DIR} with dim {vectors.shape[1]}.")

if __name__ == "__main__":
    main()
