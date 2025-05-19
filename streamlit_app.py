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
                    st.info(f"📤 Analyzing {image_file.name}")
                    items = extract_features(image_file, palm)  # uses the fixed version from features.py
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
elif st.session_state.pending_features is not None:
    confirmed_features = []
    features = st.session_state.pending_features
    uploaded_images = st.session_state.form_inputs.get("uploaded_images", [])
    
    # Divide features evenly across images
    num_images = len(uploaded_images)
    features_per_image = len(features) // num_images if num_images else len(features)
    
    for img_idx, image_file in enumerate(uploaded_images):
        start = img_idx * features_per_image
        end = (img_idx + 1) * features_per_image if img_idx < num_images - 1 else len(features)
        image_features = features[start:end]
    
        with st.container():
            # Columns: Image left, features right
            col1, col2 = st.columns([1, 2], gap="large")
    
            with col1:
                st.image(image_file, width=250, caption="")
    
            with col2:
                with st.container():
                    st.markdown("### Features")
                    for feat_idx, feature in enumerate(image_features):
                        label = feature["item"]
                        description = feature["description"]
                        key = f"feature_{img_idx}_{feat_idx}_{label}"
    
                        checked = st.checkbox(f"**{label}**", value=True, key=key)
                        if checked:
                            confirmed_features.append(feature)
    
                        st.caption(description)
                        if feat_idx < len(image_features) - 1:
                            st.markdown("<hr style='margin-top: 0.25rem; margin-bottom: 0.75rem;'>", unsafe_allow_html=True)
    
            # Separator between image blocks
            st.markdown("---")



    
    # Center-aligned Finalize Entry button
    col1, col2, col3 = st.columns([1, 3, 1])  # Adjust ratios as needed
    with col2:
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

