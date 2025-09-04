# semantic_search.py
# Cosine-similarity retrieval using NumPy. No FAISS dependency.

from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
from datetime import datetime

import numpy as np

# OpenAI embeddings (for query vectors)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

EMB_DIR = Path("embeddings")
VEC_PATH = EMB_DIR / "vectors.npy"
META_PATH = EMB_DIR / "meta.jsonl"

# lazy-loaded globals
_V = None          # np.ndarray [N, D]
_VN = None         # normalized
_META: List[Dict] = []

def _load_index():
    global _V, _VN, _META
    if _V is not None and _META:
        return
    if not VEC_PATH.exists() or not META_PATH.exists():
        _V, _VN, _META = None, None, []
        return
    _V = np.load(VEC_PATH)
    if _V.size == 0:
        _V, _VN, _META = None, None, []
        return
    _VN = _V / (np.linalg.norm(_V, axis=1, keepdims=True) + 1e-12)
    _META = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                _META.append(json.loads(line))
            except Exception:
                _META.append({})
    # pad previews
    for m in _META:
        m.setdefault("filename", "unknown.txt")
        m.setdefault("chunk_id", 0)
        m.setdefault("text_preview", "")
        m.setdefault("folder", "")
        m.setdefault("file_date", None)

def _embed_query(q: str) -> np.ndarray:
    if _use_client:
        resp = _client.embeddings.create(model=EMBED_MODEL, input=[q])  # type: ignore
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
    else:
        resp = openai.Embedding.create(model=EMBED_MODEL, input=[q])  # type: ignore
        v = np.asarray(resp["data"][0]["embedding"], dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v

def _topk_from_indices(qvec: np.ndarray, indices: Iterable[int], k: int) -> List[Tuple[int, float, Dict]]:
    idx = np.fromiter(indices, dtype=int)
    if idx.size == 0:
        return []
    sims = _VN[idx] @ qvec
    top = np.argsort(-sims)[:k]
    out = []
    for j in top:
        i = int(idx[j])
        out.append((i, float(sims[j]), _META[i]))
    return out

def _all_indices() -> Iterable[int]:
    return range(len(_META))

def _date_to_dt(iso: str | None):
    if not iso:
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d")
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def search(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    _load_index()
    if _VN is None or not _META:
        return []
    qvec = _embed_query(query)
    return _topk_from_indices(qvec, _all_indices(), k)

def search_meetings(query: str, k: int = 5) -> List[Tuple[int, float, Dict]]:
    _load_index()
    if _VN is None or not _META:
        return []
    qvec = _embed_query(query)
    indices = [i for i, m in enumerate(_META) if (m.get("folder","").lower() == "meetings")]
    return _topk_from_indices(qvec, indices, k)

def search_in_date_window(query: str, start_dt: datetime, end_dt: datetime, k: int = 5) -> List[Tuple[int, float, Dict]]:
    _load_index()
    if _VN is None or not _META:
        return []
    qvec = _embed_query(query)
    indices: List[int] = []
    for i, m in enumerate(_META):
        dt = _date_to_dt(m.get("file_date"))
        if dt and (start_dt <= dt <= end_dt):
            indices.append(i)
    return _topk_from_indices(qvec, indices, k)
