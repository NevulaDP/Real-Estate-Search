import requests
import json
from utils.hf_config import HF_REPO_ID

def load_entries_from_hub(filename="property_db.json"):
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # will throw error on 403/404/etc

        # Print/log to verify it's pulling live
        st.write("✅ Successfully pulled data from Hugging Face")
        st.write(f"🔁 Found {len(existing_entries)} existing entries")

        return json.loads(response.text)
    except Exception as e:
        print("❌ Error pulling data:", e)
        return []
