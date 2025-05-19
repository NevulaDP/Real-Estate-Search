import streamlit as st
from PIL import Image
import io
import base64
import json
import google.generativeai as palm

def extract_features(image_file, _):
  {"item": "Modern Lamp", "description": "A sleek lamp with LED light."},
  {"item": "Bookshelf", "description": "Wooden bookshelf with five shelves."}
  return []

def generate_combined_text(
    title,
    short_description,
    location,
    price,
    size,
    num_bedrooms,
    num_bathrooms,
    balcony,
    parking,
    floor,
    detected_features=None
):
    balcony_text = "This apartment includes a balcony." if balcony else "This apartment does not include a balcony."
    parking_text = "It includes a parking spot." if parking else "This apartment does not include a parking spot."

    if floor == 0:
        floor_text = "The apartment is on the ground floor."
    elif floor == 1:
        floor_text = "The apartment is on the first floor."
    else:
        floor_text = f"The apartment is on the {floor}th floor."

    base_description = (
        f"{title}. {short_description}. "
        f"This property is located in {location}. "
        f"It is priced at {price} dollars and spans {size} square feet. "
        f"It includes {num_bedrooms} bedrooms and {num_bathrooms} bathrooms. "
        f"{balcony_text} {parking_text} {floor_text}"
    )

    feature_descriptions = ""
    if detected_features:
        for feature in detected_features:
            item = feature.get("item", "").strip()
            desc = feature.get("description", "").strip()
            if item and desc:
                feature_descriptions += f" {item}: {desc}"

    return f"{base_description} {feature_descriptions.strip()}"
