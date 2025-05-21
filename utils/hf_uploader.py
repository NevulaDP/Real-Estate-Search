import os
import json
from huggingface_hub import upload_file
from utils.hf_config import HF_REPO_ID, HF_TOKEN

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

def upload_json_to_hub(new_entries, filename="property_db.json"):
    # Step 1: Load existing entries
    from utils.hf_loader import load_entries_from_hub
    existing_entries = load_entries_from_hub(filename)
    print(f"🧠 Existing entries loaded: {len(existing_entries)}")  # Add this line DEBUG

    # Step 2: Merge without duplicates
    all_entries = existing_entries + new_entries if isinstance(new_entries, list) else existing_entries + [new_entries]

    # Step 3: Save locally
    with open(filename, "w") as f:
        json.dump(all_entries, f, indent=2)

    # Step 4: Upload to HF Hub
    upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
