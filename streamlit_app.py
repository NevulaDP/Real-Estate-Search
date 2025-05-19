import streamlit as st
from PIL import Image
import uuid
import google.generativeai as palm
from sentence_transformers import SentenceTransformer

from utils.features import extract_features, generate_combined_text
from utils.database import create_property_entry

# Configure Gemini
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Init session states
if "entries" not in st.session_state:
    st.session_state.entries = []

if "pending_features" not in st.session_state:
    st.session_state.pending_features = None

if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}

st.title("🏡 Property Entry Uploader")

# --- PHASE 1: Form submission & feature extraction ---
if st.session_state.pending_features is None:
    with st.form("property_form"):
        title = st.text_input("Title")
        short_description = st.text_area("Short Description")
        location = st.text_input("Location")
        price = st.number_input("Price ($)", step=1000)
        size = st.number_input("Size (sq ft)", step=10)
        num_bedrooms = st.number_input("Number of Bedrooms", step=1)
        num_bathrooms = st.number_input("Number of Bathrooms", step=1)
        balcony = st.checkbox("Balcony")
        parking = st.checkbox("Parking")
        floor = st.number_input("Floor Number", step=1)
        uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        submitted = st.form_submit_button("Submit Entry")

    if submitted and uploaded_images:
        all_features = []

        for image_file in uploaded_images:
            st.info(f"🔍 Analyzing: {image_file.name}")
            items = extract_features(image_file, palm)
            all_features.extend(items)

        # Save inputs and features for confirmation phase
        st.session_state.pending_features = all_features
        st.session_state.form_inputs = {
            "title": title,
            "short_description": short_description,
            "location": location,
            "price": price,
            "size": size,
            "num_bedrooms": num_bedrooms,
            "num_bathrooms": num_bathrooms,
            "balcony": balcony,
            "parking": parking,
            "floor": floor,
            "uploaded_images": uploaded_images
        }

        st.rerun()

# --- PHASE 2: Confirm features, then save entry ---
elif st.session_state.pending_features is not None:
    st.subheader("✅ Confirm Extracted Features")
    confirmed_features = []

    for feature in st.session_state.pending_features:
        label = f"{feature['item']}: {feature['description']}"
        if st.checkbox(label, value=True, key=label + str(uuid.uuid4())):
            confirmed_features.append(feature)

    if st.button("Finalize Entry"):
        inputs = st.session_state.form_inputs

        # Generate final description
        combined_text = generate_combined_text(
            inputs["title"], inputs["short_description"], inputs["location"],
            inputs["price"], inputs["size"], inputs["num_bedrooms"],
            inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
            detected_features=confirmed_features
        )

        embedding = model.encode(combined_text)

        entry = create_property_entry(
            inputs["title"], inputs["short_description"], inputs["location"],
            inputs["price"], inputs["size"], inputs["num_bedrooms"],
            inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
            confirmed_features, embedding, [img.name for img in inputs["uploaded_images"]]
        )
        entry["combined_text"] = combined_text

        st.session_state.entries.append(entry)

        # Reset flow
        st.session_state.pending_features = None
        st.session_state.form_inputs = {}
        st.success("✅ Entry saved.")
        st.json(entry)
