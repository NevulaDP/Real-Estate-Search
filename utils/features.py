import streamlit as st
from PIL import Image
import io
import base64
import json

def extract_features(image_file, client):
    """Extract visual features from an image using Gemini."""
    try:
        image = Image.open(image_file).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        Analyze the following image and identify key household items beneficial for buyers.
        Return a JSON array of objects with 'item' and 'description' keys.
        Image data: {img_str[:150]}... (truncated)
        """

        #st.info("📤 Analyzing your property...")
        model = client.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            contents=[prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )


        try:
            parsed = json.loads(response.text)
            return parsed
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ Failed to parse JSON: {e}")
            return []

    except Exception as e:
        st.error(f"Gemini API call failed: {e}")
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

def generate_short_text(title, short_description, location, price, size, num_bedrooms, num_bathrooms, balcony, parking, floor, detected_features):
    feature_keywords = [f['item'] for f in detected_features]
    feature_string = ", ".join(feature_keywords)

    return (
        f"{title}. {short_description} Located in {location}. "
        f"Price: ${price}. Size: {size} sq ft. {num_bedrooms} bedrooms, "
        f"{num_bathrooms} bathrooms. Floor: {floor}. "
        f"{'Includes a balcony.' if balcony else ''} {'Includes parking.' if parking else ''} "
        f"Features: {feature_string}."
    )

def generate_semantic_text(title, short_description, location=None, detected_features=None):
    parts = []

    # Clean title
    title = title.strip().rstrip(".")
    parts.append(f"{title}.")

    # Description
    parts.append(short_description.strip())

    # Location
    if location:
        parts.append(f"Situated in {location}.")

    # Features
    if detected_features:
        highlights = []
        for f in detected_features:
            item = f.get("item", "").strip()
            desc = f.get("description", "").strip()
            if item and desc:
                highlights.append(desc)
        if highlights:
            parts.append("Key features include: " + "; ".join(highlights) + ".")

    semantic = " ".join(parts)
    return semantic

