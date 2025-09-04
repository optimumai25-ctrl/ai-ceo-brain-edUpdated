# reminders_extractor.py
from __future__ import annotations
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

REM_DIR = Path("reminders")
REM_DIR.mkdir(exist_ok=True)

ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA_HELP = """\
Expected reminder format (top lines):
Title: <short title>
Tags: tag1, tag2, ...
ValidFrom: YYYY-MM-DD
ValidTo: YYYY-MM-DD        # optional
Body: <one or a few lines of the actual fact/policy/decision>
"""

def _sanitize_filename(s: str, maxlen: int = 60) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
    return s[:maxlen] if s else "untitled"

def _coerce_iso(d: str | None) -> str | None:
    if not d: return None
    d = d.strip()
    if ISO_DATE.match(d):
        return d
    # try to parse common human dates
    for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

def _parse_structured_block(text: str) -> dict:
    """
    Parse a reminder block. Accepts either:
    1) Fully structured with Title/Tags/ValidFrom[/ValidTo]/Body
    2) Free text after 'REMINDER:' → we wrap it as Body with synthetic Title/Tags
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]

    # If it's free text ("REMINDER: Some fact..."), wrap it:
    if not any(l.lower().startswith("title:") for l in lines):
        body = re.sub(r"^REMINDER:\s*", "", text.strip(), flags=re.IGNORECASE)
        return {
            "Title": body[:60] or "Untitled",
            "Tags": ["reminder"],
            "ValidFrom": datetime.now().strftime("%Y-%m-%d"),
            "ValidTo": None,
            "Body": body,
            "_auto_structured": True,
        }

    # Structured parsing
    out = {"Title": "", "Tags": [], "ValidFrom": None, "ValidTo": None, "Body": ""}
    buf_body = []
    for l in lines:
        low = l.lower()
        if low.startswith("title:"):
            out["Title"] = l.split(":", 1)[1].strip()
        elif low.startswith("tags:"):
            tags = l.split(":", 1)[1].strip()
            out["Tags"] = [t.strip() for t in re.split(r"[;,]", tags) if t.strip()]
        elif low.startswith("validfrom:"):
            out["ValidFrom"] = _coerce_iso(l.split(":", 1)[1])
        elif low.startswith("validto:"):
            out["ValidTo"] = _coerce_iso(l.split(":", 1)[1])
        elif low.startswith("body:"):
            buf_body.append(l.split(":", 1)[1].lstrip())
        else:
            # part of body after "Body:" or free lines below header
            buf_body.append(l)
    out["Body"] = "\n".join(buf_body).strip()
    return out

def _validate(rem: dict) -> list[str]:
    errs = []
    if not rem.get("Title"):
        errs.append("Title is required.")
    if not rem.get("Tags"):
        errs.append("Tags is required (comma separated).")
    if not rem.get("ValidFrom"):
        errs.append("ValidFrom is required (YYYY-MM-DD).")
    if rem.get("ValidTo") and not ISO_DATE.match(rem["ValidTo"]):
        errs.append("ValidTo must be YYYY-MM-DD if present.")
    if not rem.get("Body"):
        errs.append("Body is required.")
    return errs

def save_reminder_block(rem: dict, ts_hint: str = "") -> Path:
    """Write a validated reminder as a .txt in reminders/ with structured header."""
    title_for_file = _sanitize_filename(rem["Title"])
    ts = ts_hint or datetime.now().strftime("%Y-%m-%d_%H%M")
    fname = f"{ts}_{title_for_file}.txt"
    fp = REM_DIR / fname
    header = [
        f"Title: {rem['Title']}",
        f"Tags: {', '.join(rem['Tags'])}",
        f"ValidFrom: {rem['ValidFrom']}",
    ]
    if rem.get("ValidTo"):
        header.append(f"ValidTo: {rem['ValidTo']}")
    header.append("Body: ")
    payload = "\n".join(header) + "\n" + rem["Body"].strip() + "\n"
    fp.write_text(payload, encoding="utf-8")
    return fp

def extract_from_csv(csv_path: str, role_col="role", content_col="content", ts_col="timestamp") -> list[Path]:
    """
    Read chat history CSV, find rows where content starts with 'REMINDER:',
    validate/structure them, and write reminders/*.txt
    """
    df = pd.read_csv(csv_path)
    out_files: list[Path] = []

    for _, row in df.iterrows():
        role = str(row.get(role_col, "")).lower()
        content = str(row.get(content_col, "")).strip()
        if role != "user":
            continue
        if not content.lower().startswith("reminder:"):
            continue

        ts = str(row.get(ts_col, "")).replace(" ", "_").replace(":", "-")
        rem = _parse_structured_block(content)

        errs = _validate(rem)
        if errs:
            # Auto-fix minimal cases for speed
            if not rem.get("Title"):
                rem["Title"] = rem["Body"][:60] or "Untitled"
            if not rem.get("Tags"):
                rem["Tags"] = ["reminder"]
            if not rem.get("ValidFrom"):
                rem["ValidFrom"] = datetime.now().strftime("%Y-%m-%d")
            # Re-validate
            errs = _validate(rem)
        if errs:
            raise ValueError("Reminder validation failed:\n" + "\n".join(f"- {e}" for e in errs) + "\n\n" + SCHEMA_HELP)

        fp = save_reminder_block(rem, ts_hint=ts)
        out_files.append(fp)

    return out_files

if __name__ == "__main__":
    # Example:
    # python reminders_extractor.py  (then change INPUT path below or call extract_from_csv from app)
    path = "chat_history.csv"
    files = extract_from_csv(path)
    print(f"Extracted {len(files)} reminders into {REM_DIR.resolve()}")
