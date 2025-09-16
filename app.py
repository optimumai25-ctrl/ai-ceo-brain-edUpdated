# app.py — FastAPI wrapper for your RAG stack
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import Lock

from answer_with_rag import answer
import file_parser
import embed_and_store

HAS_CURATOR = Path("knowledge_curator.py").exists()

app = FastAPI(title="AI CEO Assistant API")
_refresh_lock = Lock()

def bootstrap_data_dirs():
    # Cloud Run will mount a GCS bucket at /data; link local folders to it
    base = Path(os.getenv("DATA_DIR", "/data"))
    base.mkdir(parents=True, exist_ok=True)
    for name in ["reminders", "parsed_data", "embeddings"]:
        tgt = base / name
        tgt.mkdir(parents=True, exist_ok=True)
        link = Path(name)
        if not link.exists():
            link.symlink_to(tgt, target_is_directory=True)

bootstrap_data_dirs()

class AskReq(BaseModel):
    query: str
    k: int | None = 7
    restrict_to_meetings: bool = False
    use_rag: bool = True

class ReminderReq(BaseModel):
    content: str
    title_hint: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(req: AskReq):
    try:
        resp = answer(
            req.query,
            k=req.k or 7,
            restrict_to_meetings=req.restrict_to_meetings,
            use_rag=req.use_rag,
            chat_history=[],
        )
        return {"answer": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reminder")
def add_reminder(req: ReminderReq):
    from datetime import datetime
    import re
    d = Path("reminders"); d.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    title = (req.title_hint or req.content.strip().split("\n", 1)[0][:60] or "Untitled").strip()
    safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", title) or "Untitled"
    fp = d / f"{ts}_{safe}.txt"
    is_structured = any(h in req.content for h in ["Title:", "Tags:", "ValidFrom:", "Body:"])
    payload = (req.content.strip() + "\n") if is_structured else (
        f"Title: {title}\nTags: reminder\nValidFrom: {datetime.now():%Y-%m-%d}\nBody: {req.content.strip()}\n"
    )
    fp.write_text(payload, encoding="utf-8")
    return {"saved": str(fp)}

@app.post("/refresh")
def refresh():
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Refresh already in progress")
    try:
        if HAS_CURATOR:
            import knowledge_curator; knowledge_curator.main()
        file_parser.main()
        embed_and_store.main()
        return {"status": "refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _refresh_lock.release()
