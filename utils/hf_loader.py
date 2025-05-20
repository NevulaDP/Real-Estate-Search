import requests
from utils.hf_config import HF_REPO_ID

def load_entries_from_hub(filename="property_db.json"):
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return []
