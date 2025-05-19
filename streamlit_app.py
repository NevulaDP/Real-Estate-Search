import streamlit as st
from PIL import Image
import uuid
import google.generativeai as palm
from sentence_transformers import SentenceTransformer

from utils.features import extract_features, generate_combined_text
from utils.database import create_property_entry

# Configure Gemini API key
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load the embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Initialize session storage for entries
if "entries" not in st.session_state:
    st.session_state.entries = []

st.title("🏡 Property Entry Uploader")

# --- Entry Form ---
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

# --- On Submit ---
if submitted and uploaded_images:
    st.info("Processing images with Gemini...")

    all_features = []

    for image_file in uploaded_images:
        st.info(f"🔍 Analyzing: {image_file.name}")
        items = extract_features(image_file, palm)

        # Let user confirm detected features
        for item in items:
            label = f"✅ {item['item']}: {item['description']}"
            if st.checkbox(label, value=True, key=item['item'] + str(uuid.uuid4())):
                all_features.append(item)

    # Generate combined property description
    final_text = generate_combined_text(
        title, short_description, location, price, size,
        num_bedrooms, num_bathrooms, balcony, parking, floor,
        detected_features=all_features
    )

    st.success("✅ Combined Description Generated")
    st.subheader("📄 Description")
    st.write(final_text)

    # Create embedding
    embedding = model.encode(final_text)

    # Create entry object
    entry = create_property_entry(
        title, short_description, location, price, size,
        num_bedrooms, num_bathrooms, balcony, parking, floor,
        all_features, embedding, [img.name for img in uploaded_images]
    )
    entry["combined_text"] = final_text  # Optional

    # Store entry in memory (session)
    st.session_state.entries.append(entry)

    st.success("📌 Entry Saved (Session Only)")
    st.text(f"Total entries: {len(st.session_state.entries)}")
    st.json(entry)
