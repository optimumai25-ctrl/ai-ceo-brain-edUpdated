from typing import List, Dict, Tuple
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
MAX_CONTEXT_CHARS = 8000

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _merge_dedup(a: List[Tuple[int, float, Dict]],
                 b: List[Tuple[int, float, Dict]],
                 limit: int) -> List[Tuple[int, float, Dict]]:
    """
    Stable merge by vector id, keep order, cap to limit.
    Ensures date-window results (Meetings) and general results (Reminders/others) are blended.
    """
    seen, out = set(), []
    for lst in (a, b):
        for rid, dist, meta in lst:
            if rid in seen:
                continue
            out.append((rid, dist, meta))
            seen.add(rid)
            if len(out) >= limit:
                return out
    return out[:limit]

def build_context(topk: List[Tuple[int, float, Dict]]) -> str:
    """
    Compact context: [SOURCE: filename | CHUNK: id] + snippet.
    """
    parts, total = [], 0
    for _, _, meta in topk:
        fname = meta.get("filename", "unknown.txt")
        cid = meta.get("chunk_id", 0)
        text = meta.get("text_preview", "")
        snippet = f"[SOURCE: {fname} | CHUNK: {cid}]\n{text}\n"
        if total + len(snippet) > MAX_CONTEXT_CHARS:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────
# Date-window resolution (supports “this week/quarter”, “Q3 2025”, etc.)
# ─────────────────────────────────────────────────────────────
_MONTHS = "(january|february|march|april|may|june|july|august|september|october|november|december)"
_Q_PAT = re.compile(r"\bq([1-4])\s*(?:[-/ ]?\s*)?(20\d{2})\b", re.I)

def _quarter_bounds(q: int, year: int):
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
      - 'this week', 'last week'  (Mon..Sun; 'this week' ends at now)
      - 'this month', 'last month'
      - 'this quarter'
      - 'Q1 2025', 'Q3-2025', 'Q4/2026'
      - 'YYYY-MM-DD'
      - 'September 2, 2025'
    Returns (start_dt, end_dt) or None.
    """
    s = q.lower().strip()
    today = datetime.now()

    if "this week" in s:
        monday = today - timedelta(days=today.weekday())
        return (monday.replace(hour=0, minute=0, second=0, microsecond=0),
                today.replace(hour=23, minute=59, second=59, microsecond=0))

    if "last week" in s:
        weekday = today.weekday()
        last_sun = today - timedelta(days=weekday + 1)
        last_mon = last_sun - timedelta(days=6)
        return (last_mon.replace(hour=0, minute=0, second=0, microsecond=0),
                last_sun.replace(hour=23, minute=59, second=59, microsecond=0))

    if "this month" in s:
        first = today.replace(day=1)
        return (first.replace(hour=0, minute=0, second=0, microsecond=0),
                today.replace(hour=23, minute=59, second=59, microsecond=0))

    if "last month" in s:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        first_prev = last_prev.replace(day=1)
        return (first_prev.replace(hour=0, minute=0, second=0, microsecond=0),
                last_prev.replace(hour=23, minute=59, second=59, microsecond=0))

    if "this quarter" in s:
        m = today.month
        qn = 1 if m <= 3 else 2 if m <= 6 else 3 if m <= 9 else 4
        start, end = _quarter_bounds(qn, today.year)
        end = min(end, today.replace(hour=23, minute=59, second=59, microsecond=0))
        return (start, end)

    qm = _Q_PAT.search(s)
    if qm:
        qn = int(qm.group(1)); year = int(qm.group(2))
        return _quarter_bounds(qn, year)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        y, mo, d = map(int, m.groups())
        start = datetime(y, mo, d, 0, 0, 0)
        end = datetime(y, mo, d, 23, 59, 59)
        return (start, end)

    m2 = re.search(rf"{_MONTHS}\s+(\d{{1,2}}),\s*(\d{{4}})", s, re.I)
    if m2:
        month_name, dd, yy = m2.groups()
        dt = datetime.strptime(f"{month_name} {dd} {yy}", "%B %d %Y")
        return (dt.replace(hour=0, minute=0, second=0, microsecond=0),
                dt.replace(hour=23, minute=59, second=59, microsecond=0))
    return None

# ─────────────────────────────────────────────────────────────
# Generative intent bypass
# ─────────────────────────────────────────────────────────────
_GEN_PAT = re.compile(
    r"\b(idea|ideas|brainstorm|suggest|suggestions|plan|plans|strategy|strategies|framework|outline|"
    r"write|draft|improve|optimi[sz]e|approach|roadmap|design|architecture|concept|"
    r"marketing campaign|growth experiment)\b", re.I,
)
def is_generative(q: str) -> bool:
    return bool(_GEN_PAT.search(q))

# ─────────────────────────────────────────────────────────────
# Chat Completion (no temperature for GPT-5)
# ─────────────────────────────────────────────────────────────
def ask_gpt(query: str, context: str = "", chat_history: List[Dict] = [], structure: str = "none") -> str:
    system = (
        "You are a precise Virtual CEO assistant. "
        "When sources are provided, use them and cite [filename#chunk] like [2025-09-02_Meeting-Summary.txt#2]. "
        "When no sources are provided, answer with your general knowledge clearly and concisely."
    )
    if structure == "meeting_summary":
        system += ("\nFormat the answer with these exact H2 sections in order: "
                   "## Agenda\n## Decisions\n## Action Items\n"
                   "Under Action Items, list bullets as: '- Task — Owner — Due Date' if present.")

    messages: List[Dict] = [{"role": "system", "content": system}]
    for msg in chat_history[-4:]:
        content = msg.get("content", ""); ts = msg.get("timestamp", ""); role = msg.get("role", "user")
        messages.append({"role": role, "content": (f"[{ts}] {content}" if ts else content)})

    messages.append({"role": "user", "content": f"Query:\n{query}\n\nSources:\n{context}"} if context
                    else {"role": "user", "content": query})

    if _use_client:
        resp = _client.chat.completions.create(model=COMPLETIONS_MODEL, messages=messages[-6:])  # type: ignore
        return resp.choices[0].message.content
    else:
        resp = openai.ChatCompletion.create(model=COMPLETIONS_MODEL, messages=messages[-6:])  # type: ignore
        return resp.choices[0].message["content"]

# ─────────────────────────────────────────────────────────────
# Public API — BLENDED retrieval (Meetings + Reminders)
# ─────────────────────────────────────────────────────────────
def answer(
    query: str,
    k: int = 5,
    chat_history: List[Dict] = [],
    restrict_to_meetings: bool = False,
    use_rag: bool = True,
) -> str:
    """
    - Date windows blend: top date-window (Meetings) + top general (Reminders/others).
    - Meetings toggle acts as a *boost*, not a hard filter.
    - Still bypasses retrieval for generative asks or when use_rag=False.
    """
    if not use_rag or is_generative(query):
        return ask_gpt(query, context="", chat_history=chat_history, structure="none")

    def _is_meeting(meta: Dict) -> bool:
        return (meta.get("folder", "") or "").lower() == "meetings"

    # 1) If query has a date-window → BLEND
    win = resolve_date_window_from_query(query)
    if win:
        start, end = win
        date_hits = search_in_date_window(query, start, end, k=max(k, 5))  # meetings w/ dates
        gen_hits  = search(query, k=max(k, 5))                              # reminders + everything
        hits = _merge_dedup(date_hits, gen_hits, limit=k)
    else:
        # 2) No window → prefer Meetings if toggle ON, but BLEND with general
        if restrict_to_meetings:
            meet_hits = search_meetings(query, k=max(k, 5))
            gen_hits  = search(query, k=max(k, 10))
            hits = _merge_dedup(meet_hits, gen_hits, limit=k)
        else:
            hits = search(query, k=k)

    if not hits:
        return ask_gpt(query, context="", chat_history=chat_history, structure="none")

    ctx = build_context(hits)
    is_meeting_ctx = any(_is_meeting(meta) for _, _, meta in hits)
    wants_summary = bool(re.search(r"\b(summary|summarize|decisions?|action items?)\b", query, re.I))
    structure = "meeting_summary" if (is_meeting_ctx and (restrict_to_meetings or wants_summary)) else "none"
    return ask_gpt(query, context=ctx, chat_history=chat_history, structure=structure)

if __name__ == "__main__":
    print(answer("Who is our AI Coordinator this week?", k=7, restrict_to_meetings=True))
