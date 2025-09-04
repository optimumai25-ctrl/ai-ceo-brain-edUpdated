import os
import io
import csv
from pathlib import Path

import streamlit as st
import docx
import pandas as pd
from PyPDF2 import PdfReader
import fitz  # PyMuPDF

# Optional Google Drive auth — safely skipped if secrets not present
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2 import service_account
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    gdrive_secrets = st.secrets.get("gdrive", None)
    if gdrive_secrets:
        creds = service_account.Credentials.from_service_account_info(dict(gdrive_secrets), scopes=SCOPES)
        service = build("drive", "v3", credentials=creds)
    else:
        service = None
except Exception:
    service = None  # no Drive available

KB_FOLDER_NAME = "AI_CEO_KnowledgeBase"   # optional; parsed if Drive is configured
REMINDERS_FOLDER_NAME = "AI_CEO_Reminders" # optional; parsed if Drive is configured
OUTPUT_DIR = Path("parsed_data")
OUTPUT_DIR.mkdir(exist_ok=True)

LOWTEXT_LOG = OUTPUT_DIR / "low_text_files.csv"

# ─────────────────────────────────────────────────────────────
# Drive helpers (only used if 'service' is available)
# ─────────────────────────────────────────────────────────────
def list_folder_contents(parent_id):
    results = service.files().list(
        q=f"'{parent_id}' in parents and trashed = false",
        fields='files(id, name, mimeType)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    return results.get('files', [])

def get_folder_id_by_exact_name(folder_name):
    results = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed = false",
        spaces='drive',
        fields='files(id, name)',
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    folders = results.get('files', [])
    if not folders:
        raise Exception(f"Folder '{folder_name}' not found in Drive.")
    return folders[0]['id']

def download_file(file_id):
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# ─────────────────────────────────────────────────────────────
# Extractors
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(fh: io.BytesIO) -> str:
    data = fh.read()
    doc = fitz.open(stream=data, filetype="pdf")
    pages = [p.get_text("text") or "" for p in doc]
    text = "\n".join(pages)
    if len(text.strip()) < 200:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    return text

def extract_text_from_docx(fh: io.BytesIO) -> str:
    d = docx.Document(fh)
    return "\n".join([p.text for p in d.paragraphs])

def extract_text_from_excel(fh: io.BytesIO) -> str:
    df = pd.read_excel(fh)
    return df.to_string(index=False)

# ─────────────────────────────────────────────────────────────
# Save parsed TXT with headers
# ─────────────────────────────────────────────────────────────
def write_parsed_output(folder_label: str, name: str, text: str):
    base_name = os.path.splitext(name)[0].replace(' ', '_')
    output_path = OUTPUT_DIR / f"{base_name}.txt"
    output_path.write_text(f"[FOLDER]: {folder_label}\n[FILE]: {name}\n\n{text}", encoding="utf-8")
    print(f"✅ Saved to {output_path}")
    if len(text.strip()) < 500:
        hdr_exists = LOWTEXT_LOG.exists()
        with open(LOWTEXT_LOG, "a", newline="", encoding="utf-8") as lf:
            w = csv.writer(lf)
            if not hdr_exists:
                w.writerow(["folder","file","chars"])
            w.writerow([folder_label, name, len(text.strip())])

# ─────────────────────────────────────────────────────────────
# Process Drive file (if Drive is configured)
# ─────────────────────────────────────────────────────────────
def process_and_save_drive(file, folder_label):
    file_id = file['id']
    name = file['name']
    mime = file['mimeType']
    print(f"📄 Processing: {name}")
    try:
        if mime == 'application/pdf':
            fh = download_file(file_id)
            text = extract_text_from_pdf(fh)
        elif mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            fh = download_file(file_id)
            text = extract_text_from_docx(fh)
        elif mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            fh = download_file(file_id)
            text = extract_text_from_excel(fh)
        else:
            print(f"❌ Skipping unsupported file type: {name}")
            return
        write_parsed_output(folder_label, name, text)
    except Exception as e:
        print(f"❌ Error processing {name}: {e}")

# ─────────────────────────────────────────────────────────────
# Parse local reminders (*.txt) → parsed_data
# ─────────────────────────────────────────────────────────────
def parse_local_reminders():
    folder = Path("reminders")
    if not folder.exists():
        return
    for fp in folder.glob("*.txt"):
        text = fp.read_text(encoding="utf-8")
        write_parsed_output("Reminders", fp.name, text)

# ─────────────────────────────────────────────────────────────
# Parse Drive sources (optional)
# ─────────────────────────────────────────────────────────────
def parse_knowledgebase_drive():
    parent_id = get_folder_id_by_exact_name(KB_FOLDER_NAME)
    for folder in list_folder_contents(parent_id):
        if folder['mimeType'] != 'application/vnd.google-apps.folder':
            continue
        label = folder['name']
        print(f"\n📁 Scanning KB subfolder: {label}")
        for file in list_folder_contents(folder['id']):
            process_and_save_drive(file, label)

def parse_reminders_drive():
    try:
        rem_id = get_folder_id_by_exact_name(REMINDERS_FOLDER_NAME)
    except Exception:
        print(f"⚠️ Drive reminders folder '{REMINDERS_FOLDER_NAME}' not found; skipping.")
        return
    print(f"\n📁 Scanning Drive Reminders")
    for file in list_folder_contents(rem_id):
        process_and_save_drive(file, "Reminders")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    print("🔎 Parsing sources into parsed_data/*.txt …")
    # Always parse local reminders first
    parse_local_reminders()

    # Optionally parse Google Drive if configured
    if service is not None:
        try:
            parse_knowledgebase_drive()
        except Exception as e:
            print(f"⚠️ Skipping Drive KnowledgeBase due to error: {e}")
        try:
            parse_reminders_drive()
        except Exception as e:
            print(f"⚠️ Skipping Drive Reminders due to error: {e}")

    print("✅ Parsing complete.")

if __name__ == '__main__':
    main()
