# utils/database.py
import json
import numpy as np

def create_property_entry(
    property_id,
    title, short_description, location, price, size,
    num_bedrooms, num_bathrooms, balcony, parking, floor,
    detected_features, embedding, image_urls
):
    return {
        'id': property_id,
        'image_paths': image_urls,
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
        'combined_text': None,
        'short_text': None,
        'semantic_text': None
    }

