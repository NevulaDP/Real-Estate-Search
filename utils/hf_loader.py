"""
hf_loader.py

Loads the property database from a JSON file hosted on the Hugging Face Hub.
Used by both the upload and search sections.
"""

import requests
import json
from huggingface_hub import hf_hub_download
from utils.hf_config import HF_REPO_ID, HF_TOKEN
import streamlit as st

def load_entries_from_hub(filename="property_db.json"):

    """
    Downloads and loads property entries from a Hugging Face dataset repo.

    Args:
        filename (str): The name of the JSON file in the dataset repo.

    Returns:
        List[dict]: Parsed property records from the JSON file.
                    Returns an empty list if the file is missing or fails to load.
    """

    try:
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=filename,
            token=HF_TOKEN  # if needed
        )
        with open(local_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"❌ Failed to load property DB: {e}")
        return []
