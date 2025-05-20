import json

def upload_image_to_hub(image_file, property_uuid, save_dir="temp_images"):
    import os
    from huggingface_hub import upload_file
    from utils.hf_config import HF_REPO_ID, HF_TOKEN

    # Save to a temporary local file
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, image_file.name)

    with open(file_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Upload to HF with UUID in path
    hf_path = f"images/{property_uuid}/{image_file.name}"
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=hf_path,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    return f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{hf_path}"

def upload_json_to_hub(entries, filename="property_db.json"):
    with open(filename, "w") as f:
        json.dump(entries, f)

    upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )
