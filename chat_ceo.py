# chat_ceo.py
# AI CEO Assistant — Render-ready Streamlit app
# -------------------------------------------------------------
# FEATURES
# - Always-on friendly: all data written under DATA_DIR (env) or "."
# - Robust env/secrets handling for APP_USER/APP_PASS and Google SA
# - Modes: New Chat, View History, Edit Conversation, Refresh Data
# - REMINDER: prefix in chat creates structured reminder files
# - Refresh Data runs file_parser.main() + embed_and_store.main()
# - Safe fallbacks if optional modules are missing
# -------------------------------------------------------------

import os
import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Optional local modules (handle missing gracefully)
# Ensure these files exist in your repo root:
#   - file_parser.py          (must expose main())
#   - embed_and_store.py      (must expose main())
#   - answer_with_rag.py      (must expose answer(prompt, ...))
# If names differ, change the imports below accordingly.
# ─────────────────────────────────────────────────────────────
try:
    import file_parser  # type: ignore
except Exception as _e:
    file_parser = None

try:
    import embed_and_store  # type: ignore
except Exception as _e:
    embed_and_store = None

try:
    from answer_with_rag import answer  # type: ignore
except Exception as _e:
    # Simple fallback if your RAG module isn't present yet
    def answer(prompt, k=7, chat_history=None, restrict_to_meetings=False, use_rag=True):
        return (
            "RAG module not available. Echoing your message:\n\n"
            f"> {prompt}\n\n"
            "Add answer_with_rag.py with an `answer()` function for real responses."
        )

# ─────────────────────────────────────────────────────────────
# App Config (Streamlit)
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI CEO Assistant 🧠", page_icon="🧠", layout="wide")

# ─────────────────────────────────────────────────────────────
# Helpers: robust secrets/env access
# ─────────────────────────────────────────────────────────────
def _get_from_st_secrets(key, default=None):
    """Safely read from st.secrets; return default if not present."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

def get_cred(key, default=None):
    """Environment-first; fallback to st.secrets (lowercase key); then default."""
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    return _get_from_st_secrets(key.lower(), default)

def load_gdrive_service_account():
    """
    Construct a Google service account JSON dict from environment variables,
    with a fallback to st.secrets['gdrive'].
    """
    blob = os.getenv("GDRIVE_SA_JSON")
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass

    # Assemble from individual env fields (or from secrets.gdrive)
    gsecrets = _get_from_st_secrets("gdrive", {}) or {}
    pk = (os.getenv("GDRIVE_PRIVATE_KEY", "") or gsecrets.get("private_key", "")).replace("\\n", "\n")

    assembled = {
        "type": os.getenv("GDRIVE_TYPE") or gsecrets.get("type"),
        "project_id": os.getenv("GDRIVE_PROJECT_ID") or gsecrets.get("project_id"),
        "private_key_id": os.getenv("GDRIVE_PRIVATE_KEY_ID") or gsecrets.get("private_key_id"),
        "private_key": pk,
        "client_email": os.getenv("GDRIVE_CLIENT_EMAIL") or gsecrets.get("client_email"),
        "client_id": os.getenv("GDRIVE_CLIENT_ID") or gsecrets.get("client_id"),
        "auth_uri": os.getenv("GDRIVE_AUTH_URI") or gsecrets.get("auth_uri"),
        "token_uri": os.getenv("GDRIVE_TOKEN_URI") or gsecrets.get("token_uri"),
        "auth_provider_x509_cert_url": os.getenv("GDRIVE_AUTH_PROVIDER_X509_CERT_URL") or gsecrets.get("auth_provider_x509_cert_url"),
        "client_x509_cert_url": os.getenv("GDRIVE_CLIENT_X509_CERT_URL") or gsecrets.get("client_x509_cert_url"),
        "universe_domain": os.getenv("GDRIVE_UNIVERSE_DOMAIN") or gsecrets.get("universe_domain"),
    }

    # If clearly unconfigured, return None
    if not assembled["type"] and not assembled["client_email"]:
        return None
    return assembled

# ─────────────────────────────────────────────────────────────
# Credentials (ENV → secrets → default)
# ─────────────────────────────────────────────────────────────
USERNAME = get_cred("APP_USER", "admin123")
PASSWORD = get_cred("APP_PASS", "BestOrg123@#")
GDRIVE_SA = load_gdrive_service_account()  # available if needed elsewhere

# ─────────────────────────────────────────────────────────────
# Paths (honor DATA_DIR for Render/VPS persistent disks)
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("DATA_DIR", ".")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

HIST_PATH = BASE_DIR / "chat_history.json"
REFRESH_PATH = BASE_DIR / "last_refresh.txt"
REMINDERS_DIR = BASE_DIR / "reminders"
EMBED_DIR = BASE_DIR / "embeddings"
PARSED_DIR = BASE_DIR / "parsed_data"

HAS_CURATOR = Path("knowledge_curator.py").exists()

# ─────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────
def login():
    st.title("🔐 Login to AI CEO Assistant")
    with st.form("login_form"):
        u = st.text_input("👤 Username")
        p = st.text_input("🔑 Password", type="password")
        submitted = st.form_submit_button("➡️ Login")
        if submitted:
            if u == USERNAME and p == PASSWORD:
                st.session_state["authenticated"] = True
                st.success("✅ Login successful.")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login()
    st.stop()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_history():
    if HIST_PATH.exists():
        try:
            return json.loads(HIST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_history(history):
    try:
        HIST_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def reset_chat():
    try:
        if HIST_PATH.exists():
            HIST_PATH.unlink()
    except Exception as e:
        st.error(f"Failed to clear history: {e}")

def save_refresh_time():
    try:
        REFRESH_PATH.write_text(datetime.now().strftime("%b-%d-%Y %I:%M %p"))
    except Exception as e:
        st.error(f"Failed to update refresh time: {e}")

def load_refresh_time():
    try:
        if REFRESH_PATH.exists():
            return REFRESH_PATH.read_text()
    except Exception:
        pass
    return "Never"

def export_history_to_csv(history: list) -> bytes:
    df = pd.DataFrame(history)
    return df.to_csv(index=False).encode("utf-8")

def save_reminder_local(content: str, title_hint: str = "") -> str:
    """
    Save a REMINDER as a structured .txt in DATA_DIR/reminders and return the file path.
    Accepts either a plain sentence or a structured block with Title/Tags/ValidFrom/Body.
    """
    REMINDERS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    title = (title_hint or content.strip().split("\n", 1)[0][:60] or "Untitled").strip()
    safe_title = re.sub(r"[^A-Za-z0-9_\-]+", "_", title) or "Untitled"

    fp = REMINDERS_DIR / f"{ts}_{safe_title}.txt"

    is_structured = bool(re.search(r"(?mi)^\s*Title:|^\s*Tags:|^\s*ValidFrom:|^\s*Body:", content))
    if is_structured:
        payload = content.strip() + "\n"
    else:
        payload = (
            f"Title: {title}\n"
            f"Tags: reminder\n"
            f"ValidFrom: {datetime.now():%Y-%m-%d}\n"
            f"Body: {content.strip()}\n"
        )

    fp.write_text(payload, encoding="utf-8")
    return str(fp)

def _preview_snippet(text: str, limit: int = 80) -> str:
    """Make a one-line preview without using backslashes in f-string exprs."""
    return " ".join((text or "").splitlines())[:limit]

# ─────────────────────────────────────────────────────────────
# History editing helpers
# ─────────────────────────────────────────────────────────────
def update_turn(idx: int, new_content: str) -> bool:
    history = load_history()
    if idx < 0 or idx >= len(history):
        return False
    history[idx]["content"] = new_content
    history[idx]["edited_at"] = datetime.now().isoformat(timespec="seconds")
    save_history(history)
    return True

def regenerate_reply_for_user_turn(idx: int, limit_meetings: bool, use_rag: bool) -> str:
    """
    Rebuild the assistant reply for the chosen user turn.
    - Uses chat_history up to that user turn (inclusive).
    - Replaces the next assistant turn if it exists, else inserts a new one.
    """
    history = load_history()
    if idx < 0 or idx >= len(history):
        raise IndexError("Turn index out of range.")
    if history[idx].get("role") != "user":
        raise ValueError("Select a USER turn to regenerate the assistant reply.")

    ctx = history[: idx + 1]

    # Call into your RAG function; handle signature variance
    try:
        reply = answer(
            history[idx]["content"],
            k=7,
            chat_history=ctx,
            restrict_to_meetings=limit_meetings,
            use_rag=use_rag,
        )
    except TypeError:
        reply = answer(
            history[idx]["content"],
            k=7,
            chat_history=ctx,
            restrict_to_meetings=limit_meetings,
        )

    # Find next assistant turn to replace, if any
    next_assistant = None
    for j in range(idx + 1, len(history)):
        if history[j].get("role") == "assistant":
            next_assistant = j
            break
        if history[j].get("role") == "user":
            break

    ts = datetime.now().strftime("%b-%d-%Y %I:%M%p")
    if next_assistant is not None:
        history[next_assistant]["content"] = reply
        history[next_assistant]["timestamp"] = ts
        history[next_assistant]["regenerated_from_idx"] = idx
    else:
        history.insert(
            idx + 1,
            {"role": "assistant", "content": reply, "timestamp": ts, "regenerated_from_idx": idx},
        )

    save_history(history)
    return reply

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
st.sidebar.title("🧠 AI CEO Panel")
st.sidebar.markdown(f"👥 Logged in as: `{USERNAME}`")

with st.sidebar.expander("📊 Index health (embeddings)"):
    try:
        report_path = EMBED_DIR / "embedding_report.csv"
        if report_path.exists():
            df = pd.read_csv(report_path)
            st.caption(f"🧾 Rows: {len(df)}")
            if set(["chunks", "chars"]).issubset(df.columns):
                bad = df[(df["chunks"] == 0) | (df["chars"] < 200)]
                if len(bad):
                    st.warning(f"⚠️ {len(bad)} file(s) look sparse (<200 chars or 0 chunks).")
            st.dataframe(df.tail(50), use_container_width=True, height=220)
        else:
            st.caption("ℹ️ No report yet. Run **Refresh Data**.")
    except Exception as e:
        st.caption(f"ℹ️ Could not read embedding report: {e}")

with st.sidebar.expander("🧹 Curate & Restack", expanded=False):
    if not HAS_CURATOR:
        st.caption("Add `knowledge_curator.py` to enable curation.")
    else:
        if st.button("Run Curator → Rebuild Index"):
            try:
                import knowledge_curator  # type: ignore
                knowledge_curator.main()
                if file_parser is None or embed_and_store is None:
                    raise RuntimeError("file_parser or embed_and_store not available.")
                file_parser.main()
                embed_and_store.main()
                save_refresh_time()
                st.success("Curation + restack complete.")
            except Exception as e:
                st.error(f"Failed: {e}")

if st.sidebar.button("🔓 Logout"):
    st.session_state["authenticated"] = False
    st.rerun()

mode = st.sidebar.radio(
    "🧭 Navigation",
    ["💬 New Chat", "📜 View History", "✏️ Edit Conversation", "🔁 Refresh Data"],
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Start a message with **REMINDER:** to teach the assistant instantly.")

# ─────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────
if mode == "🔁 Refresh Data":
    st.title("🔁 Refresh AI Knowledge Base")
    st.caption("Parses local reminders + (optional) Google Drive docs, then re-embeds.")

    # Show where we are writing/reading
    st.markdown(f"**DATA_DIR:** `{BASE_DIR}`")
    st.markdown(f"**Reminders dir:** `{REMINDERS_DIR}`")
    st.markdown(f"**Parsed dir:** `{PARSED_DIR}`")
    st.markdown(f"**Embeddings dir:** `{EMBED_DIR}`")

    # Ensure directories exist
    for p in [REMINDERS_DIR, PARSED_DIR, EMBED_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    # Quick counts
    rem_ct = len(list(REMINDERS_DIR.glob("*.txt")))
    parsed_ct = len(list(PARSED_DIR.glob("*.txt")))
    report_path = EMBED_DIR / "embedding_report.csv"
    report_exists = report_path.exists()

    c1, c2, c3 = st.columns(3)
    c1.metric("Local REMINDER files", rem_ct)
    c2.metric("Parsed .txt files", parsed_ct)
    c3.metric("Has embedding_report.csv", "Yes" if report_exists else "No")

    # Helper button: create a demo REMINDER so you can test end-to-end
    if st.button("📝 Create demo REMINDER"):
        demo = (
            "Title: Q4 KPIs\n"
            "Tags: reminder, okr\n"
            f"ValidFrom: {datetime.now():%Y-%m-%d}\n"
            "Body: Revenue target 10% QoQ, NPS ≥ 60, ship AI dashboard by Nov 15.\n"
        )
        path = save_reminder_local(demo, title_hint="Q4_KPIs")
        st.success(f"Demo reminder saved at: {path}")

    # Run pipelines
    disabled = (file_parser is None or embed_and_store is None)
    if disabled:
        st.warning("`file_parser.py` and/or `embed_and_store.py` not found. Add them to your repo.")
    if st.button("🚀 Run File Parser + Embedder", disabled=disabled):
        with st.spinner("Refreshing knowledge base..."):
            try:
                # 1) parse reminders (and optional Drive) into PARSED_DIR
                file_parser.main()

                # Recount parsed files so we know if there is anything to embed
                parsed_ct = len(list(PARSED_DIR.glob("*.txt")))
                if parsed_ct == 0:
                    st.warning("No parsed files found. Create a REMINDER (or enable Drive) and run again.")
                else:
                    # 2) embed into EMBED_DIR
                    embed_and_store.main()

                save_refresh_time()
                st.success("✅ Data refreshed.")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.markdown(f"Last Refreshed: **{load_refresh_time()}**")

    # If a report exists, show a peek so you can verify
    if report_path.exists():
        try:
            df = pd.read_csv(report_path)
            st.caption(f"🧾 Embedding report rows: {len(df)}")
            st.dataframe(df.tail(20), use_container_width=True, height=240)
        except Exception as e:
            st.warning(f"Found report but failed to read it: {e}")
    else:
        st.info("No embedding report yet. Create a REMINDER and run the parser + embedder.")

elif mode == "📜 View History":
    st.title("📜 Chat History")
    history = load_history()
    if not history:
        st.info("No chat history found.")
    else:
        for turn in history:
            role = "👤 You" if turn.get("role") == "user" else "🧠 Assistant"
            timestamp = turn.get("timestamp", "N/A")
            st.markdown(f"**{role} | [{timestamp}]**  \n{turn.get('content', '')}")

        st.markdown("---")
        st.download_button(
            label="Download Chat History as CSV",
            data=export_history_to_csv(history),
            file_name="chat_history.csv",
            mime="text/csv",
        )
        if st.button("Clear Chat History"):
            reset_chat()
            st.success("History cleared.")

elif mode == "✏️ Edit Conversation":
    st.title("✏️ Edit Conversation")
    history = load_history()
    if not history:
        st.info("No chat history found.")
    else:
        options = [
            f"{i}: {turn.get('role','?')} | [{turn.get('timestamp','N/A')}] | {_preview_snippet(turn.get('content',''))}"
            for i, turn in enumerate(history)
        ]
        sel = st.selectbox("Select a turn to edit", options, index=0)
        idx = int(sel.split(":", 1)[0])
        turn = history[idx]

        st.caption(f"Role: {turn.get('role','?')} | Timestamp: {turn.get('timestamp','N/A')}")
        edited = st.text_area("Content", value=turn.get("content", ""), height=220)

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Save changes"):
                if update_turn(idx, edited):
                    st.success("Saved.")
                else:
                    st.error("Failed to save changes.")

        with col2:
            if turn.get("role") == "user":
                if st.button("Regenerate assistant reply from here"):
                    try:
                        reply = regenerate_reply_for_user_turn(
                            idx,
                            limit_meetings=st.session_state.get("limit_meetings", False),
                            use_rag=st.session_state.get("use_rag", True),
                        )
                        st.info("Assistant reply regenerated (updated history).")
                        st.markdown(reply)
                    except Exception as e:
                        st.error(f"Failed to regenerate: {e}")
            else:
                st.caption("Regeneration is available only for USER turns.")

        with col3:
            if turn.get("role") == "user":
                if st.button("Convert this turn to a REMINDER file"):
                    path = save_reminder_local(
                        edited,
                        title_hint=(edited.strip().split("\n", 1)[0][:60] if edited.strip() else "Reminder"),
                    )
                    st.success(f"Saved reminder: {path}. Use 'Refresh Data' to index it.")
            else:
                st.caption("Only USER turns can be converted to a REMINDER.")

elif mode == "💬 New Chat":
    st.title("🧠 AI CEO Assistant")
    st.caption("Ask about meetings, projects, policies. Start a message with REMINDER: to teach facts.")
    st.markdown(f"Last Refreshed: **{load_refresh_time()}**")

    # Persisted defaults for toggles (Meetings OFF, RAG ON)
    if "limit_meetings" not in st.session_state:
        st.session_state["limit_meetings"] = False
    if "use_rag" not in st.session_state:
        st.session_state["use_rag"] = True

    colA, colB = st.columns([1, 1])
    with colA:
        st.checkbox(
            "Limit retrieval to Meetings",
            value=st.session_state["limit_meetings"],
            key="limit_meetings",
        )
    with colB:
        st.checkbox(
            "Use internal knowledge (RAG)",
            value=st.session_state["use_rag"],
            key="use_rag",
        )

    # Show prior turns
    history = load_history()
    for turn in history:
        with st.chat_message(turn.get("role", "assistant")):
            st.markdown(f"[{turn.get('timestamp', 'N/A')}]  \n{turn.get('content', '')}")

    # Chat input
    user_msg = st.chat_input("Type your question or add a REMINDER…")
    if user_msg:
        # 1) If this is a REMINDER, save it immediately to DATA_DIR/reminders
        if user_msg.strip().lower().startswith("reminder:"):
            body = re.sub(r"^reminder:\s*", "", user_msg.strip(), flags=re.I)
            title_hint = body.split("\n", 1)[0][:60]
            saved_path = save_reminder_local(body, title_hint=title_hint)
            st.success(f"Reminder saved: `{saved_path}`. Run Refresh Data to index it.")

        # 2) Normal chat flow
        now = datetime.now().strftime("%b-%d-%Y %I:%M%p")
        history.append({"role": "user", "content": user_msg, "timestamp": now})

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                try:
                    reply = answer(
                        user_msg,
                        k=7,
                        chat_history=history,
                        restrict_to_meetings=st.session_state["limit_meetings"],
                        use_rag=st.session_state["use_rag"],
                    )
                except TypeError:
                    # Backward-compatible signature
                    reply = answer(
                        user_msg,
                        k=7,
                        chat_history=history,
                        restrict_to_meetings=st.session_state["limit_meetings"],
                    )
                except Exception as e:
                    reply = f"Error: {e}"
            ts = datetime.now().strftime("%b-%d-%Y %I:%M%p")
            st.markdown(f"[{ts}]  \n{reply}")

        history.append({"role": "assistant", "content": reply, "timestamp": ts})
        save_history(history)
