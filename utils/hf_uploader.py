"""
hf_uploader.py

Handles uploading images and property entries to the Hugging Face Hub.
Uses HF_TOKEN and HF_REPO_ID from secure configuration.

Functions:
- upload_image_to_hub: Uploads an image file to a specific repo path.
- upload_json_to_hub: Appends a new property to the JSON database on HF.
"""

import os
import json
from huggingface_hub import upload_file
from utils.hf_config import HF_REPO_ID, HF_TOKEN
import streamlit as st

def upload_image_to_hub(image_file, property_uuid, save_dir="temp_images"):

    """
    Saves an image locally and uploads it to the Hugging Face dataset repository.

    Args:
        image_file: UploadedFile object from Streamlit (or similar).
        property_uuid (str): Unique identifier to organize image under.
        save_dir (str): Local temporary directory to save the file before upload.

    Returns:
        str: Public HF URL to access the uploaded image.
    """

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, image_file.name)

    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())

    hf_path = f"images/{property_uuid}/{image_file.name}"
    
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
    return f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{hf_path}"

def upload_json_to_hub(new_entry, filename="property_db.json"):

    """
    Appends a new property entry to an existing JSON database hosted on HF.

    Args:
        new_entry (dict): The new property record to append.
        filename (str): Local filename and target HF path for the JSON database.

    Returns:
        None. Uploads modified file to HF repo.
    """

    from utils.hf_loader import load_entries_from_hub

    existing_entries = load_entries_from_hub(filename)
    all_entries = existing_entries + [new_entry]

    with open(filename, "w") as f:
        json.dump(all_entries, f, indent=2)

    upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
