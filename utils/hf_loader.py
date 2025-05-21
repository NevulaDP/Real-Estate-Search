import requests
import json
from utils.hf_config import HF_REPO_ID

def load_entries_from_hub(filename="property_db.json"):
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"❌ Failed to fetch: {response.status_code}")
            print(f"URL tried: {url}")
            return []

        try:
            return response.json()
        except json.JSONDecodeError as e:
            print("❌ JSON decoding failed.")
            print("Raw content:")
            print(response.text[:500])  # Print the beginning of file
            return []

    except Exception as e:
        print(f"❌ General exception: {e}")
        return []
