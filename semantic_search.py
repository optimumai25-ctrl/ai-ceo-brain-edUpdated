import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import os

import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

# Embedding for query
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

INDEX_PATH = Path("embeddings/faiss.index")
META_PATH = Path("embeddings/metadata.pkl")

def _embed_query_client(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)

def _embed_query_legacy(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)  # type: ignore
    return np.asarray(resp["data"][0]["embedding"], dtype=np.float32)

def embed_query(text: str) -> np.ndarray:
    arr = _embed_query_client(text) if _use_client else _embed_query_legacy(text)
    if arr.shape != (EMBED_DIM,):
        raise ValueError(f"Unexpected embedding shape {arr.shape}")
    return arr

def load_resources():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Missing FAISS index or metadata. Run embed_and_store.py first.")
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    index, metadata = load_resources()
    qvec = embed_query(query).reshape(1, -1)
    D, I = index.search(qvec, max(k, 50))
    out: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1: continue
        out.append((int(idx), float(dist), metadata.get(int(idx), {})))
    return out

def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s: return None
    try: return datetime.strptime(s, "%Y-%m-%d")
    except Exception: return None

def _query_tags(query: str) -> List[str]:
    toks = [t.strip(",.?:;!()[]").lower() for t in query.split()]
    vocab = {"hr","hiring","recruiting","finance","budget","expense","policy","product","engineering","data","sales","ops","legal","org","roles","ai","coordinator"}
    return [t for t in toks if t in vocab]

def rerank(results: List[Tuple[int, float, Dict]], query: str, prefer_meetings: bool = False, prefer_recent: bool = False) -> List[Tuple[int,float,Dict]]:
    qtags = set(_query_tags(query))
    now = datetime.now()

    def score(item):
        _, dist, meta = item
        base = -dist  # smaller distance → larger score
        folder = str(meta.get("folder","")).lower()

        # Meetings recency
        meet_date = _parse_iso(meta.get("meeting_date"))
        meet_bonus = (meet_date.toordinal()*10) if (prefer_recent and meet_date) else 0
        folder_bonus = 1000 if (prefer_meetings and folder == "meetings") else 0

        # Reminders: tag overlap + validity
        tags = set((meta.get("tags") or []))
        tag_overlap = len(qtags & {t.lower() for t in tags})
        tag_bonus = tag_overlap * 500

        vfrom = _parse_iso(meta.get("valid_from"))
        vto = _parse_iso(meta.get("valid_to"))
        valid_now = True
        if vfrom and now < vfrom: valid_now = False
        if vto and now > vto: valid_now = False
        validity_bonus = 0 if valid_now else -1000

        return folder_bonus*1_000_000 + meet_bonus + tag_bonus + validity_bonus + base

    return sorted(results, key=score, reverse=True)

def search_meetings(query: str, k: int = 5, prefer_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    raw = search(query, k=max(k, 100))
    re_ranked = rerank(raw, query=query, prefer_meetings=True, prefer_recent=prefer_recent)
    return re_ranked[:k]

def filter_by_date_range(results: List[Tuple[int, float, Dict]], start: datetime, end: datetime) -> List[Tuple[int, float, Dict]]:
    kept: List[Tuple[int, float, Dict]] = []
    for rid, dist, meta in results:
        d = _parse_iso(meta.get("meeting_date"))
        if d and start <= d <= end:
            kept.append((rid, dist, meta))
    return kept

def rerank_for_recency(results: List[Tuple[int, float, Dict]], query: str, favor_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    return rerank(results, query=query, prefer_meetings=False, prefer_recent=favor_recent)

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    pool = search(query, k=max(k, 200))
    windowed = filter_by_date_range(pool, start, end)
    if not windowed:
        return []
    return rerank_for_recency(windowed, query=query)[:k]

if __name__ == "__main__":
    hits = search_meetings("hr hiring policy last month", k=5)
    for i, (vid, dist, meta) in enumerate(hits, 1):
        print(f"{i}. dist={dist:.4f} file={meta.get('filename')} folder={meta.get('folder')} tags={meta.get('tags')} valid_from={meta.get('valid_from')} valid_to={meta.get('valid_to')}")
        print(meta.get("text_preview","")[:160], "\n---")
