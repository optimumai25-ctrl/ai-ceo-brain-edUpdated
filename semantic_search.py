import os
import pickle
from typing import List, Tuple, Dict, Callable, Optional
from datetime import datetime
import numpy as np

# FAISS
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

# OpenAI embeddings
try:
    from openai import OpenAI
    _eclient = OpenAI()
    _use_eclient = True
except Exception:
    _eclient = None
    _use_eclient = False
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Paths
DATA_DIR = os.getenv("DATA_DIR", ".")
EMB_DIR = os.path.join(DATA_DIR, "embeddings")
FAISS_PATH = os.path.join(EMB_DIR, "index.faiss")
META_PATH = os.path.join(EMB_DIR, "metadata.pkl")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Cache
_index = None
_meta: List[Dict] = []
_dim = None
_ip_index = False  # whether index is inner-product

# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────
def _load_index_and_meta():
    global _index, _meta, _dim, _ip_index
    if _index is not None and _meta:
        return

    if faiss is None:
        raise RuntimeError("faiss is not installed. Add faiss-cpu to requirements.txt.")

    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata not found at {META_PATH}")

    _index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        _meta = pickle.load(f)
    # Infer metric
    _dim = _index.d if hasattr(_index, "d") else None
    try:
        _ip_index = isinstance(_index, faiss.IndexFlatIP) or _index.metric_type == faiss.METRIC_INNER_PRODUCT
    except Exception:
        _ip_index = False

# ─────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    text = text.replace("\n", " ").strip()
    if _use_eclient:
        v = _eclient.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding  # type: ignore
    else:
        v = openai.Embedding.create(model=EMBEDDING_MODEL, input=[text])["data"][0]["embedding"]  # type: ignore
    vec = np.array(v, dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-12) if _ip_index else vec

# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(dt_str[:19], fmt)
        except Exception:
            continue
    return None

def _in_range(dt: Optional[datetime], start: datetime, end: datetime) -> bool:
    if not dt:
        return False
    return start <= dt <= end

def _default_filter(_: Dict) -> bool:
    return True

# ─────────────────────────────────────────────────────────────
# Core search + rerank
# ─────────────────────────────────────────────────────────────
def _search_core(query: str,
                 k: int,
                 filter_fn: Callable[[Dict], bool] = _default_filter,
                 prefer_meetings: bool = False,
                 prefer_recent: bool = False) -> List[Tuple[int, float, Dict]]:
    _load_index_and_meta()
    qv = _embed(query).reshape(1, -1)

    # raw search
    D, I = _index.search(qv, min(k * 8, max(32, k * 4)))  # ask for more, then filter/rerank
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx < 0 or idx >= len(_meta):
            continue
        meta = _meta[idx]
        if not filter_fn(meta):
            continue
        results.append((idx, float(dist), meta))

    if not results:
        return []

    # rerank
    results = rerank(results, query, prefer_meetings=prefer_meetings, prefer_recent=prefer_recent)
    return results[:k]

def rerank(results: List[Tuple[int, float, Dict]],
           query: str,
           prefer_meetings: bool = False,
           prefer_recent: bool = False) -> List[Tuple[int, float, Dict]]:
    """
    Gentle preferences:
      - Meetings get a small boost if requested.
      - Newer meeting_date gets a small boost if prefer_recent.
      - ValidTo in the past gets a penalty.
    We do NOT swamp the base similarity (keeps Reminders competitive).
    """
    now = datetime.now()

    def score(item, rank0: int):
        _, dist, meta = item
        base = 1000 - rank0  # stable tie-break

        folder = str(meta.get("folder", "")).lower()
        folder_bonus = 50 if (prefer_meetings and folder == "meetings") else 0  # gentle

        meet_dt = _parse_iso(meta.get("meeting_date"))
        recency_bonus = 0
        if prefer_recent and meet_dt:
            days_ago = (now - meet_dt).days
            recency_bonus = max(0, 200 - min(200, days_ago))  # 0..200

        valid_from = _parse_iso(meta.get("valid_from"))
        valid_to = _parse_iso(meta.get("valid_to"))
        validity_bonus = 0
        if valid_from and valid_from > now:
            validity_bonus -= 100
        if valid_to and valid_to < now:
            validity_bonus -= 200

        return base + folder_bonus + recency_bonus + validity_bonus

    scored = [(score(item, rnk), item) for rnk, item in enumerate(results)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored]

# ─────────────────────────────────────────────────────────────
# Public APIs
# ─────────────────────────────────────────────────────────────
def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    """General semantic search across all folders (Reminders, Meetings, Finance, etc.)."""
    return _search_core(query, k=k, filter_fn=_default_filter, prefer_meetings=False, prefer_recent=False)

def search_meetings(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    """Prefer Meetings. Still allows others if caller blends separately."""
    def filt(meta: Dict) -> bool:
        return True
    return _search_core(query, k=k, filter_fn=filt, prefer_meetings=True, prefer_recent=True)

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    """
    Date-scoped search: focuses on items with a 'meeting_date' within [start, end].
    Typically Meetings; Reminders lack dates, so they won't appear here.
    (answer_with_rag.py blends these with general search so Reminders are not ignored.)
    """
    def filt(meta: Dict) -> bool:
        dt = _parse_iso(meta.get("meeting_date"))
        return _in_range(dt, start, end)
    return _search_core(query, k=k, filter_fn=filt, prefer_meetings=True, prefer_recent=True)


