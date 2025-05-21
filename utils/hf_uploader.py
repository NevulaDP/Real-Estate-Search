import os
import json
from huggingface_hub import upload_file
from utils.hf_config import HF_REPO_ID, HF_TOKEN
import streamlit as st

def upload_image_to_hub(image_file, property_uuid, save_dir="temp_images"):
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
    try:
        from utils.hf_loader import load_entries_from_hub

        # Load entries directly from Hugging Face
        existing_entries = load_entries_from_hub(filename)

        # Avoid duplicates
        if not any(e["id"] == new_entry["id"] for e in existing_entries):
            existing_entries.append(new_entry)

        # Save to local file
        with open(filename, "w") as f:
            json.dump(existing_entries, f, indent=2)

        # Upload to Hugging Face
        upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        st.error(f"❌ Failed to upload property DB: {e}")
