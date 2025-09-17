import os
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# ==============================
# Google Drive Auth
# ==============================
SCOPES = ["https://www.googleapis.com/auth/drive"]
gdrive_secrets = st.secrets["gdrive"]

credentials = service_account.Credentials.from_service_account_info(
    dict(gdrive_secrets), scopes=SCOPES
)
service = build("drive", "v3", credentials=credentials)

# If you are using a Shared Drive
SHARED_DRIVE_ID = gdrive_secrets.get("shared_drive_id", None)


# ==============================
# Folder Logic
# ==============================
def find_or_create_folder(service, name, parent_id=None):
    query = (
        f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' "
        f"and trashed = false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        driveId=SHARED_DRIVE_ID if SHARED_DRIVE_ID else None,
        corpora="drive" if SHARED_DRIVE_ID else "user"
    ).execute()

    folders = results.get("files", [])
    if folders:
        return folders[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(
        body=metadata,
        fields="id",
        supportsAllDrives=True
    ).execute()

    return folder["id"]


# ==============================
# Upload or Update File
# ==============================
def upload_or_update_file(service, file_path, folder_id):
    file_name = os.path.basename(file_path)
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed = false"

    results = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()

    files = results.get("files", [])
    media = MediaFileUpload(file_path, resumable=True)

    if files:
        # Update existing file
        file_id = files[0]["id"]
        service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True
        ).execute()
        print(f"🔁 Updated: {file_name}")
    else:
        # Upload new file
        metadata = {"name": file_name, "parents": [folder_id]}
        service.files().create(
            body=metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True
        ).execute()
        print(f"✅ Uploaded: {file_name}")
