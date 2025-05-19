import streamlit as st
from PIL import Image
import io
import base64
import json
import google.generativeai as palm

def extract_features(image_file, _):
    try:
        image = Image.open(image_file).convert("RGB")
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = f"""
        You are a visual assistant. Return a JSON array of household items in the image.
        Each object must have 'item' and 'description'. No explanation — JSON only.

        Image (base64): {img_str[:200]}... (truncated for brevity)
        """

        # Log that we are generating
        st.write("🟡 Sending prompt to Gemini")

        model = palm.GenerativeModel(model_name="gemini-pro")
        response = model.generate_content(prompt)

        # Log the raw response
        st.write("🧠 Gemini raw response:", response.text)

        return json.loads(response.text)

    except json.JSONDecodeError:
        st.warning("⚠️ Could not parse Gemini response as JSON.")
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
