# gdrive_uploader.py
# My Drive by default; supports Shared Drives if you set `shared_drive_id` in st.secrets["gdrive"].

import os
from typing import Optional, Tuple

import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# ==============================
# Google Drive Auth
# ==============================
SCOPES = ["https://www.googleapis.com/auth/drive"]

def get_service() -> Tuple[object, Optional[str]]:
    """
    Returns:
        (service, shared_drive_id)
        - service: Google Drive v3 client
        - shared_drive_id: str if configured, else None (My Drive)
    """
    gdrive_secrets = st.secrets["gdrive"]
    creds = service_account.Credentials.from_service_account_info(
        dict(gdrive_secrets), scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)
    shared_drive_id = gdrive_secrets.get("shared_drive_id", None)
    return service, shared_drive_id


# ==============================
# Internal helpers
# ==============================
def _list_files(service, q: str, drive_id: Optional[str] = None, fields: str = "files(id, name, mimeType, parents)"):
    """
    List files/folders using the correct corpora and drive scope.
    - My Drive: corpora="user"
    - Shared Drive: corpora="drive" with driveId
    """
    if drive_id:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields=fields,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            corpora="drive",
            driveId=drive_id,
        ).execute()
    else:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields=fields,
            corpora="user",
        ).execute()
    return resp.get("files", [])


def _default_parent_id(drive_id: Optional[str]) -> str:
    """
    For My Drive, root is 'root'.
    For Shared Drives, the root folder id equals the shared drive id.
    """
    return drive_id if drive_id else "root"


# ==============================
# Folder Logic
# ==============================
def find_or_create_folder(service, name: str, parent_id: Optional[str] = None, drive_id: Optional[str] = None) -> str:
    """
    Ensure a folder exists and return its id.

    Args:
        service: Drive v3 client
        name: folder name to find or create
        parent_id: parent folder id; if None, uses drive root (My Drive root or Shared Drive root)
        drive_id: shared drive id if using a Shared Drive, else None

    Returns:
        folder_id (str)
    """
    parent_id = parent_id or _default_parent_id(drive_id)

    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and name='{name}' and '{parent_id}' in parents and trashed=false"
    )
    found = _list_files(service, query, drive_id)
    if found:
        return found[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }

    if drive_id:
        folder = service.files().create(
            body=metadata,
            fields="id",
            supportsAllDrives=True,
        ).execute()
    else:
        folder = service.files().create(
            body=metadata,
            fields="id",
        ).execute()

    return folder["id"]


# ==============================
# Upload or Update File
# ==============================
def upload_or_update_file(service, file_path: str, folder_id: str, drive_id: Optional[str] = None) -> str:
    """
    Upload a file to a folder, or update if a file with the same name already exists.

    Args:
        service: Drive v3 client
        file_path: local path to file
        folder_id: destination folder id
        drive_id: shared drive id if using a Shared Drive, else None

    Returns:
        file_id (str) of the uploaded/updated file
    """
    file_name = os.path.basename(file_path)
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    existing = _list_files(service, query, drive_id, fields="files(id, name)")

    media = MediaFileUpload(file_path, resumable=True)

    if existing:
        file_id = existing[0]["id"]
        # Update existing file
        if drive_id:
            service.files().update(
                fileId=file_id,
                media_body=media,
                supportsAllDrives=True,
            ).execute()
        else:
            service.files().update(
                fileId=file_id,
                media_body=media,
            ).execute()
        print(f"Updated: {file_name}")
        return file_id

    # Create new file
    metadata = {"name": file_name, "parents": [folder_id]}
    if drive_id:
        created = service.files().create(
            body=metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
    else:
        created = service.files().create(
            body=metadata,
            media_body=media,
            fields="id",
        ).execute()

    print(f"Uploaded: {file_name}")
    return created["id"]


# ==============================
# Convenience: one-shot upload
# ==============================
def upload_csv_to_path(file_path: str, *folders: str) -> str:
    """
    Convenience helper:
    1) Builds service + resolves My Drive or Shared Drive context.
    2) Walks/creates the folder path (e.g., "AI_CEO_KnowledgeBase", "Chat_History").
    3) Uploads/updates the file.

    Usage:
        upload_csv_to_path("chat_history.csv", "AI_CEO_KnowledgeBase", "Chat_History")
    """
    service, drive_id = get_service()

    parent_id = _default_parent_id(drive_id)
    for name in folders:
        parent_id = find_or_create_folder(service, name, parent_id=parent_id, drive_id=drive_id)

    return upload_or_update_file(service, file_path, parent_id, drive_id=drive_id)


