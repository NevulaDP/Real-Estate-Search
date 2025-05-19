# utils/database.py
import json
import uuid
import numpy as np

def create_property_entry(
    title, short_description, location, price, size,
    num_bedrooms, num_bathrooms, balcony, parking, floor,
    detected_features, embedding, image_filenames
):
    return {
        'id': str(uuid.uuid4()),
        'image_paths': image_filenames,
        'title': title,
        'short_description': short_description,
        'location': location,
        'price': price,
        'size': size,
        'num_bedrooms': num_bedrooms,
        'num_bathrooms': num_bathrooms,
        'balcony': balcony,
        'parking': parking,
        'floor': floor,
        'detected_features': detected_features,
        'embedding': embedding.tolist(),
        'combined_text': None  # Optional, you can store it here too
    }

def save_entries_to_file(entries, filename='embeddings.json'):
    with open(filename, 'w') as f:
        json.dump(entries, f)

def load_entries_from_file(filename='embeddings.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
