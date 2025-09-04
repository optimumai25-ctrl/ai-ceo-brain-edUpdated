from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import re
import os

from semantic_search import (
    search,
    search_meetings,
    search_in_date_window,
)

# OpenAI client setup (new SDK preferred, fallback legacy)
try:
    from openai import OpenAI
    _client = OpenAI()
    _use_client = True
except Exception:
    _client = None
    _use_client = False
    import openai  # type: ignore
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Use GPT-5 for chat/answers
COMPLETIONS_MODEL = "gpt-5"

# ─────────────────────────────────────────────────────────────
# Context Builder (fast-mode aware)
# ─────────────────────────────────────────────────────────────
def build_context(
    topk: List[Tuple[int, float, Dict]],
    total_max_chars: int,
    per_snippet_max: int,
) -> str:
    """
    Compact context with per-snippet and total caps.
    Format: [SOURCE: filename | CHUNK: id]\n<snippet>\n
    """
    parts, total = [], 0
    for _, _, meta in topk:
        fname = meta.get("filename", "unknown.txt")
        cid = meta.get("chunk_id", 0)
        text = (meta.get("text_preview", "") or "")[:per_snippet_max]
        snippet = f"[SOURCE: {fname} | CHUNK: {cid}]\n{text}\n"
        if total + len(snippet) > total_max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────
# Date-window resolution from user query (extended)
# ─────────────────────────────────────────────────────────────
_MONTHS = "(january|february|march|april|may|june|july|august|september|october|november|december)"
_Q_PAT = re.compile(r"\bq([1-4])\s*(?:[-/ ]?\s*)?(20\d{2})\b", re.I)  # Q1 2025 / Q3-2025 / Q4/2026

def _quarter_bounds(q: int, year: int):
    # Q1: Jan–Mar; Q2: Apr–Jun; Q3: Jul–Sep; Q4: Oct–Dec
    starts = {1: (1, 1), 2: (4, 1), 3: (7, 1), 4: (10, 1)}
    sm, sd = starts[q]
    if q < 4:
        em, ed = starts[q + 1]
        end = datetime(year, em, ed) - timedelta(days=1)
    else:
        end = datetime(year, 12, 31)
    start = datetime(year, sm, sd)
    return (
        start.replace(hour=0, minute=0, second=0, microsecond=0),
        end.replace(hour=23, minute=59, second=59, microsecond=0),
    )

def resolve_date_window_from_query(q: str):
    """
    Recognize:
      - 'this week', 'last week'  (weeks are Mon..Sun; 'this week' ends at now)
      - 'this month', 'last month'
      - 'this quarter'
      - 'Q1 2025', 'Q3-2025', 'Q4/2026'
      - 'YYYY-MM-DD'
      - 'September 2, 2025'
    Returns (start_dt, end_dt) or None.
    """
    s = q.lower().strip()
    today = datetime.now()

    # week windows
    if "this week" in s:
        monday = today - timedelta(days=today.weekday())
        return (
            monday.replace(hour=0, minute=0, second=0, microsecond=0),
            today.replace(hour=23, minute=59, second=59, microsecond=0),
        )

    if "last week" in s:
        weekday = today.weekday()  # Mon=0
        last_sun = today - timedelta(days=weekday + 1)
        last_mon = last_sun - timedelta(days=6)
        return (
            last_mon.replace(hour=0, minute=0, second=0, microsecond=0),
            last_sun.replace(hour=23, minute=59, second=59, microsecond=0),
        )

    # month windows
    if "this month" in s:
        first = today.replace(day=1)
        return (
            first.replace(hour=0, minute=0, second=0, microsecond=0),
            today.replace(hour=23, minute=59, second=59, microsecond=0),
        )

    if "last month" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return (
            first_prev.replace(hour=0, minute=0, second=0, microsecond=0),
            last_prev.replace(hour=23, minute=59, second=59, microsecond=0),
        )

    # quarter windows
    if "this quarter" in s:
        m = today.month
        qn = 1 if m <= 3 else 2 if m <= 6 else 3 if m <= 9 else 4
        start, end = _quarter_bounds(qn, today.year)
        end = min(end, today.replace(hour=23, minute=59, second=59, microsecond=0))
        return (start, end)

    qm = _Q_PAT.search(s)
    if qm:
        qn = int(qm.group(1))
        year = int(qm.group(2))
        return _quarter_bounds(qn, year)

    # ISO date
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        start = datetime(y, mo, d, 0, 0, 0)
        end = datetime(y, mo, d, 23, 59, 59)
        return (start, end)

    # "September 2, 2025"
    m2 = re.search(rf"{_MONTHS}\s+(\d{{1,2}}),\s*(\d{{4}})", s, re.I)
    if m2:
        month_name, dd, yy = m2.groups()
        dt = datetime.strptime(f"{month_name} {dd} {yy}", "%B %d %Y")
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = dt.replace(hour=23, minute=59, second=59, microsecond=0)
        return (start, end)

    return None

# ─────────────────────────────────────────────────────────────
# Generative intent bypass (act like regular GPT for brainstorming)
# ─────────────────────────────────────────────────────────────
_GEN_PAT = re.compile(
    r"\b(idea|ideas|brainstorm|suggest|suggestions|plan|plans|strategy|strategies|framework|outline|"
    r"write|draft|improve|optimi[sz]e|approach|roadmap|design|architecture|concept|"
    r"marketing campaign|growth experiment)\b",
    re.I,
)
def is_generative(q: str) -> bool:
    return bool(_GEN_PAT.search(q))

# ─────────────────────────────────────────────────────────────
# Chat Completion (GPT-5: use max_completion_tokens on new SDK; no temperature)
# ─────────────────────────────────────────────────────────────
def _extract_text(resp) -> str:
    """
    Robustly extract text from Chat Completions response.
    Handles cases where content may be empty or structured.
    """
    try:
        content = resp.choices[0].message.content
        if content:
            return content
    except Exception:
        pass

    # Fallback: some SDKs might return structured content lists
    try:
        mc = getattr(resp.choices[0].message, "content", None)
        if isinstance(mc, list):
            parts = []
            for p in mc:
                if isinstance(p, dict):
                    t = p.get("text")
                    if t:
                        parts.append(t)
                elif isinstance(p, str):
                    parts.append(p)
            if parts:
                return "".join(parts)
    except Exception:
        pass

    return "[No text returned by the model.]"

def ask_gpt(
    query: str,
    context: str = "",
    chat_history: List[Dict] = [],
    structure: str = "none",
    max_completion_tokens: Optional[int] = None,
) -> str:
    system = (
        "You are a precise Virtual CEO assistant. "
        "When sources are provided, use them and cite [filename#chunk] like [2025-09-02_Meeting-Summary.txt#2]. "
        "When no sources are provided, answer with your general knowledge clearly and concisely."
    )

    if structure == "meeting_summary":
        system += (
            "\nFormat the answer with these exact H2 sections in order: "
            "## Agenda\n## Decisions\n## Action Items\n"
            "Under Action Items, list bullets as: '- Task — Owner — Due Date' if present."
        )

    messages: List[Dict] = [{"role": "system", "content": system}]

    # Use only last 2 history turns (faster)
    for msg in chat_history[-2:]:
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        role = msg.get("role", "user")
        messages.append({"role": role, "content": f"[{timestamp}] {content}" if timestamp else content})

    messages.append({"role": "user", "content": f"Query:\n{query}\n\nSources:\n{context}"} if context else {"role": "user", "content": query})

    # Build kwargs per SDK path
    if _use_client:
        kwargs = dict(model=COMPLETIONS_MODEL, messages=messages[-6:])
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        resp = _client.chat.completions.create(**kwargs)  # type: ignore
        return _extract_text(resp)
    else:
        # Legacy client: do NOT pass max_* tokens for GPT-5 (unsupported there)
        kwargs = dict(model=COMPLETIONS_MODEL, messages=messages[-6:])
        resp = openai.ChatCompletion.create(**kwargs)  # type: ignore
        return _extract_text(resp)

# ─────────────────────────────────────────────────────────────
# Public API (fallbacks A & B, fast-mode aware)
# ─────────────────────────────────────────────────────────────
def answer(
    query: str,
    k: int = 5,
    chat_history: List[Dict] = [],
    restrict_to_meetings: bool = False,
    use_rag: bool = True,
    fast_mode: bool = True,
) -> str:
    """
    - Date windows when found.
    - Skip retrieval for generative asks or when use_rag=False.
    - Structured meeting digests when appropriate.
    - Fallbacks:
        A) Meetings-only requested but no meeting hits → general search.
        B) Date-window returns no hits → general search (no window).
    - Fast mode:
        * Smaller RAG context
        * Shorter output via max_completion_tokens (new SDK path)
        * Less history
    """
    # Fast-mode caps
    total_cap = 3000 if fast_mode else 8000
    per_snip = 600 if fast_mode else 1200
    out_tokens = 500 if fast_mode else None

    # Generative bypass or explicit GPT-only mode
    if not use_rag or is_generative(query):
        return ask_gpt(query, context="", chat_history=chat_history, structure="none", max_completion_tokens=out_tokens)

    def _has_meeting_hits(hs):
        for _, _, meta in hs or []:
            if (meta.get("folder", "") or "").lower() == "meetings":
                return True
        return False

    hits: List[Tuple[int, float, Dict]] = []

    # 1) Date-scoped search if query contains a window
    win = resolve_date_window_from_query(query)
    if win:
        start, end = win
        hits = search_in_date_window(query, start, end, k=k)

        # Fallback B: date window yielded nothing → general search
        if not hits:
            hits = search(query, k=k)

        # Fallback A: forced meetings but not actually meeting hits → general search
        if restrict_to_meetings and not _has_meeting_hits(hits):
            alt = search(query, k=k)
            if alt:
                hits = alt
    else:
        # 2) No date window
        hits = search_meetings(query, k=k) if restrict_to_meetings else search(query, k=k)

        # Fallback A
        if restrict_to_meetings and not _has_meeting_hits(hits):
            alt = search(query, k=k)
            if alt:
                hits = alt

    if not hits:
        return ask_gpt(query, context="", chat_history=chat_history, structure="none", max_completion_tokens=out_tokens)

    ctx = build_context(hits, total_max_chars=total_cap, per_snippet_max=per_snip)
    is_meeting_ctx = any((meta.get("folder", "").lower() == "meetings") for _, _, meta in hits)
    wants_summary = bool(re.search(r"\b(summary|summarize|decisions?|action items?)\b", query, re.I))
    structure = "meeting_summary" if (is_meeting_ctx and (restrict_to_meetings or wants_summary)) else "none"

    return ask_gpt(query, context=ctx, chat_history=chat_history, structure=structure, max_completion_tokens=out_tokens)

# Optional CLI test
if __name__ == "__main__":
    print(answer("Who is our AI Coordinator this week?", k=4, restrict_to_meetings=True, fast_mode=True))
