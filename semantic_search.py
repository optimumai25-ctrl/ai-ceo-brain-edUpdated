import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import os

import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

# Prefer new SDK, fallback to legacy
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
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
    # search more than k to allow dedupe/ rerank later
    D, I = index.search(qvec, max(k, 100))
    out: List[Tuple[int, float, Dict]] = []
    for dist, idx in zip(D[0], I[0]):
        if int(idx) == -1:
            continue
        out.append((int(idx), float(dist), metadata.get(int(idx), {})))
    return out


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None


def _is_reminder_active(meta: Dict, start: datetime, end: datetime) -> bool:
    """
    A reminder is considered active within [start,end] if it overlaps the window.
    If no dates, treat as active.
    """
    vf = _parse_iso(meta.get("valid_from"))
    vt = _parse_iso(meta.get("valid_to"))
    if not vf and not vt:
        return True
    if vf and vf > end:
        return False
    if vt and vt < start:
        return False
    return True


def _query_tags(query: str) -> List[str]:
    toks = [t.strip(",.?:;!()[]").lower() for t in query.split()]
    vocab = {
        "hr", "hiring", "recruiting", "finance", "budget", "expense", "policy",
        "product", "engineering", "data", "sales", "ops", "legal", "org", "roles",
        "ai", "coordinator", "meeting", "agenda", "decision"
    }
    return [t for t in toks if t in vocab]


def rerank(results: List[Tuple[int, float, Dict]], query: str,
           prefer_meetings: bool = False, prefer_recent: bool = False) -> List[Tuple[int, float, Dict]]:
    """
    Score by vector similarity + folder bonus + tag overlap + validity + mild age decay.
    Then deduplicate by canonical_key (or title/filename) to keep the best version per topic.
    """
    qtags = set(_query_tags(query))
    now = datetime.now()

    def _ts(meta: Dict) -> Optional[datetime]:
        for k in ("meeting_date", "valid_from", "version_ts"):
            d = _parse_iso(meta.get(k))
            if d:
                return d
        return None

    scored = []
    for item in results:
        _, dist, meta = item
        base = -dist  # lower distance is better
        folder = str(meta.get("folder", "")).lower()

        ts = _ts(meta)
        recency = (ts.toordinal() if (prefer_recent and ts) else 0)

        folder_bonus = 3000 if (prefer_meetings and folder == "meetings") else 0

        tags = set((meta.get("tags") or []))
        tag_overlap = len(qtags & {t.lower() for t in tags})
        tag_bonus = tag_overlap * 800

        # validity now
        vfrom = _parse_iso(meta.get("valid_from"))
        vto = _parse_iso(meta.get("valid_to"))
        valid_now = True
        if vfrom and now < vfrom:
            valid_now = False
        if vto and now > vto:
            valid_now = False
        validity_bonus = 0 if valid_now else -3000

        # mild age penalty
        age_penalty = 0
        if ts:
            days = (now - ts).days
            age_penalty = -min(days, 365)  # cap the penalty

        score = folder_bonus + tag_bonus + recency + validity_bonus + base + age_penalty
        scored.append((score, item))

    # Deduplicate by canonical_key/title/filename
    best_by_key: Dict[str, Tuple[float, Tuple[int, float, Dict]]] = {}
    for score, item in scored:
        _, _, meta = item
        key = (meta.get("canonical_key") or meta.get("title") or meta.get("filename") or "").lower()
        cur = best_by_key.get(key)
        if not cur or score > cur[0]:
            best_by_key[key] = (score, item)

    out = [v[1] for v in sorted(best_by_key.values(), key=lambda x: x[0], reverse=True)]
    return out


def search_meetings(query: str, k: int = 5, prefer_recent: bool = True) -> List[Tuple[int, float, Dict]]:
    raw = search(query, k=max(k, 100))
    re_ranked = rerank(raw, query=query, prefer_meetings=True, prefer_recent=prefer_recent)
    return re_ranked[:k]


def filter_by_date_range(results: List[Tuple[int, float, Dict]], start: datetime, end: datetime) -> List[Tuple[int, float, Dict]]:
    kept: List[Tuple[int, float, Dict]] = []
    for rid, dist, meta in results:
        folder = (meta.get("folder", "") or "").lower()
        if folder == "meetings":
            d = _parse_iso(meta.get("meeting_date"))
            if d and start <= d <= end:
                kept.append((rid, dist, meta))
        else:
            if _is_reminder_active(meta, start, end):
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
    # Simple smoke test
    hits = search_meetings("budget last week", k=5)
    for i, (vid, dist, meta) in enumerate(hits, 1):
        print(f"{i}. dist={dist:.4f} file={meta.get('filename')} folder={meta.get('folder')} tags={meta.get('tags')}")
        print(meta.get("text_preview", "")[:120], "\n---")
