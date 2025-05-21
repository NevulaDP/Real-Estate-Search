import requests
import json
from huggingface_hub import hf_hub_download
from utils.hf_config import HF_REPO_ID, HF_TOKEN
import streamlit as st

def load_entries_from_hub(filename="property_db.json"):
    import time
    import random

    # Force cache busting by adding a random query param
    cache_buster = f"?nocache={time.time()}_{random.randint(0,9999)}"
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{filename}{cache_buster}"

    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        return []
