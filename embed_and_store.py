import os
import time
import pickle
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import faiss
from dotenv import load_dotenv
from tqdm import tqdm

from chunk_utils import simple_chunks

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

PARSED_DIR = Path("parsed_data")
EMBED_DIR = Path("embeddings")
EMBED_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
INDEX_PATH = EMBED_DIR / "faiss.index"
META_PATH = EMBED_DIR / "metadata.pkl"
REPORT_CSV = EMBED_DIR / "embedding_report.csv"
REG_CSV = EMBED_DIR / "knowledge_registry.csv"

_base_index = faiss.IndexFlatL2(EMBED_DIM)
_index = faiss.IndexIDMap2(_base_index)

_metadata: Dict[int, Dict] = {}
_next_id = 0

# --- Meeting filename patterns (support canonical + human-readable) ---
_CANON_MEETING = re.compile(
    r'^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_Meeting[-_ ]?Summary',
    re.IGNORECASE,
)
_HUMAN_MEETING = re.compile(
    r'^Meeting[_ -]Summary[_ -](?P<d>\d{1,2})[-_ ](?P<mon>[A-Za-z]+)[-_ ](?P<y>\d{4})',
    re.IGNORECASE,
)
_MONTHS_MAP = {
    m.lower(): i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"], start=1
    )
}
# Also allow short month keys
_MONTHS_MAP.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
})


def _date_from_filename(fname: str) -> Optional[str]:
    """
    Returns ISO date (YYYY-MM-DD) if filename encodes a meeting date in either style:
    1) 2025-09-02_Meeting-Summary...
    2) Meeting_Summary_02-September-2025 (or short month)
    """
    stem = Path(fname).stem

    m = _CANON_MEETING.match(stem)
    if m:
        try:
            return datetime(int(m["y"]), int(m["m"]), int(m["d"])).strftime("%Y-%m-%d")
        except Exception:
            pass

    h = _HUMAN_MEETING.match(stem)
    if h:
        try:
            day = int(h["d"])
            mon_token = h["mon"].lower()
            mon = _MONTHS_MAP.get(mon_token)
            year = int(h["y"])
            if mon:
                return datetime(year, mon, day).strftime("%Y-%m-%d")
        except Exception:
            pass

    return None


ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _coerce_iso(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    d = d.strip()
    if ISO_DATE.match(d):
        return d
    for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def _extract_headers(text: str) -> dict:
    """
    Read first ~25 lines of parsed file to capture:
      - [FOLDER], [FILE]
      - For 'Reminders' files: Title/Tags/ValidFrom/ValidTo if present
    """
    out = {
        "folder": "",
        "original_file": "",
        "title": "",
        "tags": [],
        "valid_from": None,
        "valid_to": None,
    }
    lines = text.splitlines()[:25]
    for ln in lines:
        if ln.startswith("[FOLDER]:"):
            out["folder"] = ln.split(":", 1)[1].strip()
        elif ln.startswith("[FILE]:"):
            out["original_file"] = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("title:"):
            out["title"] = ln.split(":", 1)[1].strip()
        elif ln.lower().startswith("tags:"):
            tags = ln.split(":", 1)[1]
            out["tags"] = [t.strip().lower() for t in re.split(r"[;,]", tags) if t.strip()]
        elif ln.lower().startswith("validfrom:"):
            out["valid_from"] = _coerce_iso(ln.split(":", 1)[1])
        elif ln.lower().startswith("validto:"):
            out["valid_to"] = _coerce_iso(ln.split(":", 1)[1])
    return out


def _embed_client(text: str) -> np.ndarray:
    resp = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.asarray(resp.data[0].embedding, dtype=np.float32)


def _embed_legacy(text: str) -> np.ndarray:
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)  # type: ignore
    return np.asarray(resp["data"][0]["embedding"], dtype=np.float32)


def get_embedding(text: str) -> Optional[np.ndarray]:
    for attempt in range(4):
        try:
            arr = _embed_client(text) if _use_client else _embed_legacy(text)
            if arr.shape != (EMBED_DIM,):
                raise ValueError(f"Unexpected embedding shape {arr.shape}")
            return arr
        except Exception as e:
            wait = 1.5 ** attempt
            print(f"Embedding error (attempt {attempt + 1}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    print("Failed to embed after retries.")
    return None


def add_to_index(vec: np.ndarray, vid: int) -> None:
    _index.add_with_ids(vec.reshape(1, -1), np.array([vid], dtype=np.int64))


def _load_registry() -> Dict[str, Dict]:
    """
    Load knowledge_registry.csv (if present) to decide keep/supersede.
    Keyed by absolute path string in the 'path' column.
    """
    reg: Dict[str, Dict] = {}
    if REG_CSV.exists():
        with open(REG_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                reg[row.get("path", "")] = row
    return reg


def main():
    global _next_id
    if not PARSED_DIR.exists():
        print(f"Missing folder: {PARSED_DIR.resolve()}")
        return

    files = sorted([p for p in PARSED_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])
    if not files:
        print("No .txt files found in parsed_data.")
        return

    registry = _load_registry()

    print(f"Found {len(files)} files to embed.")
    report_rows: List[tuple] = [(
        "filename", "folder", "meeting_date", "title", "tags", "valid_from", "valid_to",
        "canonical_key", "version_ts", "chunks", "chars"
    )]

    for fp in tqdm(files, desc="Embedding"):
        # Skip archived files if registry says so
        reg_row = registry.get(str(fp))
        if reg_row and reg_row.get("status") == "archived":
            continue

        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            print(f"Skipping empty: {fp.name}")
            continue

        headers = _extract_headers(text)
        folder_label = headers["folder"]
        orig_name = headers["original_file"] or fp.name

        meeting_date_iso = _date_from_filename(orig_name) if folder_label.lower() == "meetings" else None
        title = headers["title"]
        tags = headers["tags"]
        valid_from = headers["valid_from"]
        valid_to = headers["valid_to"]

        # Derive canonical_key and version_ts
        if reg_row:
            canonical_key = reg_row.get("canonical_key", "") or (title or Path(fp.name).stem).lower()
            version_ts = reg_row.get("version_ts", "") or (meeting_date_iso or valid_from or "")
        else:
            if folder_label.lower() == "meetings":
                canonical_key = Path(fp.name).stem.lower()
                version_ts = meeting_date_iso or ""
            else:
                canonical_key = (title or Path(fp.name).stem).lower()
                version_ts = valid_from or ""

        chunks = simple_chunks(text, max_chars=3500, overlap=300) or [{"chunk_id": 0, "text": text[:3500]}]
        total_chars = sum(len(ch["text"]) for ch in chunks)
        report_rows.append((
            fp.name, folder_label or "", meeting_date_iso or "", title, ";".join(tags),
            valid_from or "", valid_to or "", canonical_key, version_ts, len(chunks), total_chars
        ))

        for ch in chunks:
            vec = get_embedding(ch["text"])
            if vec is None:
                print(f"Skipping chunk {ch['chunk_id']} of {fp.name} due to embedding failure.")
                continue
            add_to_index(vec, _next_id)
            _metadata[_next_id] = {
                "filename": fp.name,
                "path": str(fp),
                "chunk_id": ch["chunk_id"],
                "text_preview": ch["text"][:1000],
                "folder": folder_label,
                "meeting_date": meeting_date_iso,
                "title": title,
                "tags": tags,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "canonical_key": canonical_key,
                "version_ts": version_ts,
            }
            _next_id += 1

    faiss.write_index(_index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump(_metadata, f)

    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(report_rows)

    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"Saved metadata for {len(_metadata)} vectors to {META_PATH}")
    print(f"Wrote embedding health report to {REPORT_CSV}")


if __name__ == "__main__":
    main()

