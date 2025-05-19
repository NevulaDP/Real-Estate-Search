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
