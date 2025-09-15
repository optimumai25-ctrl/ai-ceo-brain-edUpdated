from typing import List, Dict
import re

def simple_chunks(text: str, max_chars: int = 3500, overlap: int = 300) -> List[Dict]:
    """
    Split text into overlapping chunks at paragraph boundaries.
    - max_chars: soft limit per chunk
    - overlap: carry last N chars from previous chunk to next
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks, cur, size = [], [], 0
    for p in paras:
        if size + len(p) + 2 <= max_chars:
            cur.append(p); size += len(p) + 2
        else:
            if cur:
                chunks.append("\n\n".join(cur))
            tail = "\n\n".join(cur)[-overlap:] if cur and overlap > 0 else ""
            cur = [tail, p] if tail else [p]
            size = len("\n\n".join(cur))
    if cur:
        chunks.append("\n\n".join(cur))
    return [{"chunk_id": i, "text": c} for i, c in enumerate(chunks)]

