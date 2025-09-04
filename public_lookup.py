# public_lookup.py
# Minimal public web fallback for live facts/news without heavy dependencies.
# Requires only: requests, beautifulsoup4

from __future__ import annotations
import os
import re
import time
from typing import List, Dict, Optional
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

# ---------------- Config ----------------
DEFAULT_TIMEOUT = 10
MAX_RESULTS = int(os.getenv("PUBLIC_WEB_MAX_RESULTS", "3"))
USER_AGENT = os.getenv("PUBLIC_WEB_UA", "AI-CEO/1.0 (+local)")
HEADERS = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

# cache: query -> answer or results
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

# ---------------- Search (DuckDuckGo HTML) ----------------
def _ddg_html_search(query: str, max_results: int) -> List[Dict]:
    """
    Scrape DuckDuckGo's HTML results. Return [{'title','url','snippet'}].
    """
    base = "https://duckduckgo.com/html/"
    params = {
        "q": query,
        "kl": "wt-wt",  # world
        "kz": "1",      # regionless
        "kaf": "1",     # include news/web
        "ia": "web",
    }
    url = f"{base}?{urlencode(params)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
    except Exception:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for res in soup.select(".result"):
        a = res.select_one("a.result__a") or res.select_one("a.result__url")
        if not a or not a.get("href"):
            continue
        title = a.get_text(strip=True) or a.get("href")
        href = a.get("href")
        snip_el = res.select_one(".result__snippet") or res.select_one(".result__extras__url")
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        results.append({"title": title, "url": href, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results

# ---------------- Fetch & extract ----------------
def _fetch(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.text
    except Exception:
        return None

def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def web_search(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Returns: [{'title','url','snippet','text'}]
    """
    cache_key = f"search::{query}::{max_results}"
    c = _get_cached(cache_key)
    if c:
        return c["results"]

    hits = _ddg_html_search(query, max_results=max_results)
    out: List[Dict] = []
    for h in hits:
        url = h["url"]
        html = _fetch(url)
        text = _extract_text(html) if html else ""
        out.append({"title": h["title"], "url": url, "snippet": h["snippet"], "text": text})

    _set_cache(cache_key, {"results": out})
    return out

# ---------------- Summarize with GPT-5 ----------------
def _openai_client():
    try:
        from openai import OpenAI  # new SDK
        return OpenAI()
    except Exception:
        return None

def summarize_with_gpt5(query: str, docs: List[Dict]) -> str:
    """
    Answer with inline citations [1], [2] and a Sources list.
    """
    sources = []
    chunks = []
    idx = 0
    for d in docs:
        url = d["url"]
        title = d.get("title") or url
        text = d.get("text") or d.get("snippet") or ""
        if not text or len(text) < 200:
            continue
        idx += 1
        clip = text[:4000]
        chunks.append(f"### Source {idx}: {title}\nURL: {url}\n\n{clip}")
        sources.append(f"[{idx}] {title} — {url}")
        if idx >= MAX_RESULTS:
            break

    if not chunks:
        return "No reliable public content was retrievable for this query."

    context_block = "\n\n".join(chunks)

    system = (
        "You are a precise assistant. Answer using only the provided sources. "
        "Add short inline citations like [1], [2] matching the numbered sources. "
        "If sources conflict, note the disagreement briefly."
    )
    prompt = (
        f"User question:\n{query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Instructions:\n"
        "- Produce a concise answer (3–8 sentences).\n"
        "- Add inline citations [n] where n is the source index.\n"
        "- Then append a 'Sources' section listing the URLs.\n"
    )

    client = _openai_client()
    if client:
        resp = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,  # no temperature for gpt-5 in Chat Completions
        )
        return resp.choices[0].message.content

    # legacy client
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
    cache_key = f"web_answer::{query}::{max_results}"
    c = _get_cached(cache_key)
    if c:
        return c

    docs = web_search(query, max_results=max_results)
    if not docs:
        return "No reliable public sources were found to answer this query."
    answer = summarize_with_gpt5(query, docs)
    _set_cache(cache_key, answer)
    return answer
