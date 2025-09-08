import os
import pickle
from typing import List, Tuple, Dict, Callable, Optional
from datetime import datetime
import numpy as np

# FAISS
try:
    import faiss  # type: ignore
except Exception as e:
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

# Paths (aligned with your embedder output)
DATA_DIR  = os.getenv("DATA_DIR", ".")
EMB_DIR   = os.path.join(DATA_DIR, "embeddings")
FAISS_PATH = os.path.join(EMB_DIR, "faiss.index")   # <- matches: "Saved FAISS index to embeddings/faiss.index"
META_PATH  = os.path.join(EMB_DIR, "metadata.pkl")

# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

# Cache
_index = None
_meta: List[Dict] = []
_dim = None
_ip_index = False  # whether index is inner-product (cosine)

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
# Embedding
# ─────────────────────────────────────────────────────────────
def _embed(text: str) -> np.ndarray:
    text = text.replace("\n", " ").strip()
    if _use_eclient:
        v = _eclient.embeddings.create(model=EMBEDDING_MODEL, input=text).data[0].embedding  # type: ignore
    else:
        v = openai.Embedding.create(model=EMBEDDING_MODEL, input=[text])["data"][0]["embedding"]  # type: ignore
    vec = np.array(v, dtype=np.float32)
    # Normalize for cosine/IP if index is IP
    return vec / (np.linalg.norm(vec) + 1e-12) if _ip_index else vec

# ─────────────────────────────────────────────────────────────
# Loading (self-heals by creating an empty FAISS if missing)
# ─────────────────────────────────────────────────────────────
def _load_index_and_meta():
    import numpy as _np
    global _index, _meta, _dim, _ip_index
    if _index is not None and _meta:
        return

    if faiss is None:
        raise RuntimeError("faiss is not installed. Add 'faiss-cpu' to requirements.txt (Linux).")

    os.makedirs(EMB_DIR, exist_ok=True)
    index_exists = os.path.exists(FAISS_PATH)
    meta_exists  = os.path.exists(META_PATH)

    # Bootstrap empty index if not present (fresh deployment)
    if not (index_exists and meta_exists):
        # Probe dimension using current embedding model
        # (Requires OPENAI_API_KEY; if not set, the app will still run but searches will be empty.)
        try:
            probe = _embed("bootstrap-dimension-probe")
            dim = int(probe.shape[0])
        except Exception:
            # Fallback to a common dim; will be corrected on first real embed
            dim = 3072
        _ip_index = True  # we normalize vectors → use IP
        _index = faiss.IndexFlatIP(dim)
        faiss.write_index(_index, FAISS_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump([], f)
        _dim = dim
        _meta = []
        return  # empty index loaded; searches return []

    _index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        _meta = pickle.load(f)
    _dim = _index.d if hasattr(_index, "d") else None
    try:
        _ip_index = isinstance(_index, faiss.IndexFlatIP) or _index.metric_type == faiss.METRIC_INNER_PRODUCT
    except Exception:
        _ip_index = False

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

    # raw search (request more, then filter/rerank)
    D, I = _index.search(qv, min(k * 8, max(32, k * 4)))
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

    # rerank with gentle preferences
    results = rerank(results, query, prefer_meetings=prefer_meetings, prefer_recent=prefer_recent)
    return results[:k]

def rerank(results: List[Tuple[int, float, Dict]],
           query: str,
           prefer_meetings: bool = False,
           prefer_recent: bool = False) -> List[Tuple[int, float, Dict]]:
    """
    Gentle preferences so Meetings don't swamp Reminders:
      - +50 if prefer_meetings and folder == "meetings"
      - Recency bonus up to +200 if prefer_recent and meeting_date present
      - Expired ValidTo gets penalties
    """
    now = datetime.now()

    def score(item, rank0: int):
        _, dist, meta = item

        # Base monotonic tie-breaker from raw order
        base = 1000 - rank0

        folder = str(meta.get("folder", "")).lower()
        folder_bonus = 50 if (prefer_meetings and folder == "meetings") else 0

        # Recency (meetings)
        meet_dt = _parse_iso(meta.get("meeting_date"))
        recency_bonus = 0
        if prefer_recent and meet_dt:
            days_ago = (now - meet_dt).days
            recency_bonus = max(0, 200 - min(200, days_ago))  # 0..200

        # Validity window for reminders
        valid_from = _parse_iso(meta.get("valid_from"))
        valid_to = _parse_iso(meta.get("valid_to"))
        validity_bonus = 0
        if valid_from and valid_from > now:
            validity_bonus -= 100
        if valid_to and valid_to < now:
            validity_bonus -= 200  # expired reminder → downweight

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
    """Prefer Meetings (still allows others if caller blends separately)."""
    def filt(meta: Dict) -> bool:
        return True  # not hard filtering; caller may blend with general search
    return _search_core(query, k=k, filter_fn=filt, prefer_meetings=True, prefer_recent=True)

def search_in_date_window(query: str, start: datetime, end: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    """
    Date-scoped search: focuses on items with 'meeting_date' within [start, end].
    Typically Meetings; Reminders lack dates. 'answer_with_rag' blends with general search.
    """
    def filt(meta: Dict) -> bool:
        dt = _parse_iso(meta.get("meeting_date"))
        return _in_range(dt, start, end)
    return _search_core(query, k=k, filter_fn=filt, prefer_meetings=True, prefer_recent=True)
