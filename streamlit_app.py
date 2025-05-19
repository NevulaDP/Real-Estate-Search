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
    # Only execute this if the form is submitted
        if submitted and uploaded_images:
                all_features = []

                for image_file in uploaded_images:
                    st.info(f"🔍 Analyzing image: {image_file.name}")
                    items = extract_features(image_file, palm)  # uses the fixed version from features.py
                    st.write("🧪 Features from Gemini:", items)  # debug output
                    all_features.extend(items)
                
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
# --- PHASE 2: Confirm features, then save entry ---
elif st.session_state.pending_features is not None:
    st.subheader("✅ Confirm Extracted Features")
    
    confirmed_features = []
    
    uploaded_images = st.session_state.form_inputs.get("uploaded_images", [])
    
    # Layout: images on the left, features on the right
    col_img, col_features = st.columns([1, 3])
    
    with col_img:
        if uploaded_images:
           # st.markdown("### Uploaded Images")
            for img in uploaded_images:
                st.image(img, use_container_width=True)
    
    with col_features:
        for idx, feature in enumerate(st.session_state.pending_features):
            label = feature["item"]
            description = feature["description"]
            key = f"feature_{idx}_{label}"
    
            if st.checkbox(f"**{label}**", value=True, key=key):
                confirmed_features.append(feature)
    
            st.caption(description)
            st.markdown("<hr style='margin-top: 0.25rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)



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

        # Save the entry
        st.session_state.entries.append(entry)

        # Reset state for next entry
        st.session_state.pending_features = None
        st.session_state.form_inputs = {}

        st.success("✅ Entry saved.")
        st.json(entry)

