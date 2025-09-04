# public_lookup.py
# General public web search fallback for live facts/news.
# - Uses DuckDuckGo (no API key).
# - Fetches top results, extracts readable text, then asks GPT-5 to summarize with citations.
# - Whitelisted to a small k (default 3) for speed and reliability.

from __future__ import annotations
import os
import re
import time
from typing import List, Dict, Optional

import requests
from duckduckgo_search import DDGS
import trafilatura
from bs4 import BeautifulSoup

# ---------------- Config ----------------
DEFAULT_TIMEOUT = 10
MAX_RESULTS = int(os.getenv("PUBLIC_WEB_MAX_RESULTS", "3"))
USER_AGENT = os.getenv("PUBLIC_WEB_UA", "AI-CEO/1.0 (+local)")

HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

# Simple in-memory cache (query -> result) to avoid repeated calls
_CACHE: Dict[str, Dict] = {}
TTL_SECONDS = int(os.getenv("PUBLIC_WEB_TTL_SECONDS", "1800"))  # 30 min

def _get_cached(key: str) -> Optional[Dict]:
    item = _CACHE.get(key)
    if not item:
        return None
    if (time.time() - item["ts"]) > TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    return item["val"]

def _set_cache(key: str, val: Dict) -> None:
    _CACHE[key] = {"ts": time.time(), "val": val}

# ---------------- Fetch & Extract ----------------
def _fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _extract_main_text(url: str) -> str:
    """
    Prefer trafilatura (readability-quality). Fallback to minimal BeautifulSoup text.
    """
    try:
        raw = trafilatura.fetch_url(url)
        if raw:
            text = trafilatura.extract(raw, include_comments=False, include_tables=False)
            if text and len(text.strip()) > 200:
                return text.strip()
    except Exception:
        pass

    # Fallback: simple soup-based text
    html = _fetch(url)
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = text.strip()
        return text
    except Exception:
        return ""

# ---------------- Search ----------------
def web_search(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Returns list of dicts: [{'title','href','snippet','text'}]
    """
    cached = _get_cached(f"search::{query}::{max_results}")
    if cached:
        return cached["results"]

    results: List[Dict] = []
    try:
        with DDGS() as ddg:
            hits = ddg.text(
                keywords=query,
                region="wt-wt",
                safesearch="moderate",
                timelimit="y",  # prefer recent, but not too strict
                max_results=max_results,
            )
        for h in hits or []:
            url = h.get("href") or h.get("link") or ""
            if not url:
                continue
            title = h.get("title", "")
            snippet = h.get("body", "") or h.get("snippet", "")
            # Extract page text (best-effort)
            body_text = _extract_main_text(url)
            results.append({"title": title, "url": url, "snippet": snippet, "text": body_text})
    except Exception:
        pass

    _set_cache(f"search::{query}::{max_results}", {"results": results})
    return results

# ---------------- Summarize ----------------
def _openai_client():
    try:
        from openai import OpenAI  # new SDK
        return OpenAI()
    except Exception:
        # legacy
        import openai  # type: ignore
        return None

def summarize_with_gpt5(query: str, docs: List[Dict]) -> str:
    """
    Compose a concise answer and include inline citations like [1], [2],
    and append a Sources list mapping numbers to URLs.
    """
    # Build sources block
    sources = []
    chunks = []
    for i, d in enumerate(docs, 1):
        url = d["url"]
        title = d.get("title") or url
        text = d.get("text") or d.get("snippet") or ""
        if not text:
            continue
        sources.append(f"[{i}] {title} — {url}")
        # Clip to keep prompt modest
        clip = text[:4000]
        chunks.append(f"### Source {i}: {title}\nURL: {url}\n\n{clip}")

    if not chunks:
        return "I couldn’t retrieve enough content from the web to answer confidently."

    context_block = "\n\n".join(chunks)

    system = (
        "You are a precise assistant. Answer the user's question using only the provided sources. "
        "Use neutral, factual language. Include short inline citations like [1], [2] matching the URLs below. "
        "If the sources conflict, note the disagreement briefly."
    )

    prompt = (
        f"User question:\n{query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Instructions:\n"
        "- Produce a concise answer (3-8 sentences).\n"
        "- Add inline citations [n] where n is the source index.\n"
        "- Then append a 'Sources' section listing the URLs.\n"
    )

    client = _openai_client()
    if client:
        # New SDK path
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            # NOTE: GPT-5 chat does not accept non-default temperature; omit it.
            max_tokens=600,
        )
        return resp.choices[0].message.content

    # Legacy fallback client
    import openai  # type: ignore
    resp = openai.ChatCompletion.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=600,
    )
    return resp.choices[0].message["content"]

# ---------------- Public interface ----------------
def web_answer(query: str, max_results: int = MAX_RESULTS) -> str:
    """
    End-to-end: search → fetch → extract → summarize. Returns a ready-to-display answer.
    """
    cache_key = f"web_answer::{query}::{max_results}"
    cached = _get_cached(cache_key)
    if cached:
        return cached

    hits = web_search(query, max_results=max_results)
    if not hits:
        return "No reliable public sources were found to answer this query."

    answer = summarize_with_gpt5(query, hits[:max_results])
    _set_cache(cache_key, answer)
    return answer
