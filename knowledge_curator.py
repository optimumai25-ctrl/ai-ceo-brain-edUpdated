# knowledge_curator.py
from __future__ import annotations
import csv, json, re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
import os

REM_DIR = Path("reminders")
PARSED_DIR = Path("parsed_data")
REG_CSV = Path("embeddings/knowledge_registry.csv")
HIST_PATH = Path("chat_history.json")

ISO = "%Y-%m-%d"

def _coerce_iso(s: Optional[str]) -> Optional[str]:
    if not s: return None
    for fmt in (ISO, "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y"):
        try: return datetime.strptime(s.strip(), fmt).strftime(ISO)
        except Exception: pass
    return None

def _mtime_iso(p: Path) -> str:
    return datetime.fromtimestamp(p.stat().st_mtime).strftime(ISO)

def _norm_key(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    stop = {"the","a","an","for","to","of","and","in","on","at","by","with","is","are"}
    toks = [t for t in s.split() if t and t not in stop]
    return "-".join(toks[:8]) or "untitled"

@dataclass
class Item:
    canonical_key: str
    title: str
    tags: List[str]
    valid_from: Optional[str]
    valid_to: Optional[str]
    source: str          # "Reminders" | "Meetings" | folder name
    version_ts: str      # YYYY-MM-DD
    path: str
    status: str = "active"
    supersedes: str = ""
    superseded_by: str = ""

def _read_reminder_file(fp: Path) -> Optional[Item]:
    """Read reminders/*.txt (structured header)."""
    title = tags = valid_from = valid_to = None
    txt = fp.read_text(encoding="utf-8", errors="ignore")
    for line in txt.splitlines()[:40]:
        low = line.lower()
        if low.startswith("title:"):
            title = line.split(":",1)[1].strip()
        elif low.startswith("tags:"):
            tags = [t.strip().lower() for t in re.split(r"[;,]", line.split(":",1)[1]) if t.strip()]
        elif low.startswith("validfrom:"):
            valid_from = _coerce_iso(line.split(":",1)[1])
        elif low.startswith("validto:"):
            valid_to = _coerce_iso(line.split(":",1)[1])
    if not title:
        return None
    key = _norm_key(title)
    vf = valid_from or _mtime_iso(fp)
    return Item(
        canonical_key=key,
        title=title,
        tags=tags or [],
        valid_from=valid_from,
        valid_to=valid_to,
        source="Reminders",
        version_ts=vf,
        path=str(fp),
    )

def _read_parsed_headers(fp: Path) -> Dict[str,str]:
    out = {"folder":"","file":""}
    for ln in fp.read_text(encoding="utf-8", errors="ignore").splitlines()[:25]:
        if ln.startswith("[FOLDER]:"):
            out["folder"] = ln.split(":",1)[1].strip()
        elif ln.startswith("[FILE]:"):
            out["file"] = ln.split(":",1)[1].strip()
    return out

def _read_meeting_item(fp: Path) -> Optional[Item]:
    # Expect embed_and_store to set meeting_date later; here we use filename date if present
    stem = fp.stem
    # Try YYYY-MM-DD_Meeting-Summary*
    m = re.match(r'^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})_Meeting[-_ ]?Summary', stem, re.I)
    md: Optional[str] = None
    if m:
        md = f"{m['y']}-{m['m']}-{m['d']}"
    else:
        # Meeting_Summary_02-September-2025
        m2 = re.match(r'^Meeting[_ -]Summary[_ -](\d{1,2})[-_ ]([A-Za-z]+)[-_ ](\d{4})', stem, re.I)
        if m2:
            day, mon, year = m2.groups()
            try:
                md = datetime.strptime(f"{day} {mon} {year}", "%d %B %Y").strftime(ISO)
            except Exception:
                try:
                    md = datetime.strptime(f"{day} {mon} {year}", "%d %b %Y").strftime(ISO)
                except Exception:
                    md = None
    hdr = _read_parsed_headers(fp)
    if hdr.get("folder","").lower() != "meetings":
        return None
    title = hdr.get("file") or fp.name
    key = _norm_key(title)
    ver = md or _mtime_iso(fp)
    return Item(
        canonical_key=key, title=title, tags=["meeting"],
        valid_from=md or None, valid_to=None, source="Meetings",
        version_ts=ver, path=str(fp)
    )

def _mine_chat_reminders() -> List[Item]:
    """Convert chat lines starting with 'REMINDER:' into Items."""
    if not HIST_PATH.exists():
        return []
    data = json.loads(HIST_PATH.read_text(encoding="utf-8"))
    out: List[Item] = []
    for row in data:
        if str(row.get("role","")).lower() != "user":
            continue
        content = str(row.get("content","")).strip()
        if not content.lower().startswith("reminder:"):
            continue
        body = re.sub(r"^reminder:\s*", "", content, flags=re.I).strip()
        title = body.split("\n",1)[0][:60] or "Untitled"
        key = _norm_key(title)
        ts = row.get("timestamp","")[:12]  # e.g., "Sep-11-2025"
        try:
            ver = datetime.strptime(ts, "%b-%d-%Y").strftime(ISO)
        except Exception:
            ver = datetime.now().strftime(ISO)
        out.append(Item(
            canonical_key=key, title=title, tags=["reminder"], valid_from=ver,
            valid_to=None, source="Chat", version_ts=ver, path=f"chat:{ts}"
        ))
    return out

def load_items() -> List[Item]:
    items: List[Item] = []
    # 1) structured reminders
    if REM_DIR.exists():
        for fp in REM_DIR.glob("*.txt"):
            it = _read_reminder_file(fp)
            if it: items.append(it)
    # 2) parsed_data meetings (headers already written)
    if PARSED_DIR.exists():
        for fp in PARSED_DIR.glob("*.txt"):
            it = _read_meeting_item(fp)
            if it: items.append(it)
    # 3) chat-derived reminders
    items.extend(_mine_chat_reminders())
    return items

def decide(items: List[Item]) -> List[Item]:
    # group by key
    groups: Dict[str, List[Item]] = {}
    for it in items:
        groups.setdefault(it.canonical_key, []).append(it)

    decided: List[Item] = []
    for key, arr in groups.items():
        arr.sort(key=lambda x: x.version_ts, reverse=True)  # newest first
        newest = arr[0]
        newest.status = "active"
        # supersede/expire older siblings
        for old in arr[1:]:
            # keep if pinned
            if "pinned" in old.tags:
                old.status = "active"
            else:
                old.status = "archived"
                old.superseded_by = newest.title
                # if old is still open-ended valid_to, close it the day before newest
                if old.valid_to is None and newest.valid_from:
                    try:
                        dt = datetime.strptime(newest.valid_from, ISO) - timedelta(days=1)
                        old.valid_to = dt.strftime(ISO)
                    except Exception:
                        pass
            decided.append(old)
        decided.append(newest)
    return decided

def write_registry(items: List[Item]) -> None:
    REG_CSV.parent.mkdir(exist_ok=True, parents=True)
    with open(REG_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical_key","title","tags","valid_from","valid_to","source","version_ts","path","status","supersedes","superseded_by"])
        for it in items:
            w.writerow([
                it.canonical_key, it.title, ";".join(it.tags), it.valid_from or "",
                it.valid_to or "", it.source, it.version_ts, it.path, it.status,
                it.supersedes, it.superseded_by
            ])

def main():
    items = load_items()
    decided = decide(items)
    write_registry(decided)
    print(f"Curated {len(decided)} items → {REG_CSV}")

if __name__ == "__main__":
    main()
