import streamlit as st
from PIL import Image
import uuid
import google.generativeai as palm
from sentence_transformers import SentenceTransformer

from utils.features import extract_features, generate_combined_text
from utils.database import create_property_entry
from utils.hf_uploader import upload_image_to_hub, upload_json_to_hub
from utils.hf_loader import load_entries_from_hub
#from utils.form_validation import validate_and_process_form

# Configure Gemini
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

st.set_page_config(page_title="Property Matcher", layout="wide")

# Init session states
if "entries" not in st.session_state:
    st.session_state.entries = load_entries_from_hub()

if "pending_features" not in st.session_state:
    st.session_state.pending_features = None

if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}


# --- SIDEBAR ---#
st.sidebar.title("Property Matcher")
#st.sidebar.markdown("### 🔍 Navigation")
mode = st.sidebar.radio("Go to:",["🏡 Upload Property", "🔎 Search Properties"])

if mode == "🏡 Upload Property":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏡 Upload Property")
    # --- PHASE 1: Form submission & feature extraction ---
    if st.session_state.pending_features is None:
        # Centered form layout: [empty, main, empty]
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("property_form"):
            
                # Declare the button early for logic
                submitted = st.form_submit_button("hidden_submit", type="primary")
            
                # --- FORM INPUTS ---
                title = st.text_input("Title")
                if submitted and not title.strip():
                    st.warning("Title is required.", icon="⚠️")
            
                short_description = st.text_area("Short Description")
                if submitted and not short_description.strip():
                    st.warning("Short description is required.", icon="⚠️")
            
                location = st.text_input("Location")
                if submitted and not location.strip():
                    st.warning("Location is required.", icon="⚠️")
            
                price = st.number_input("Price ($)", min_value=1, step=1000)
                size = st.number_input("Size (sq ft)", min_value=1, step=10)
                num_bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
                num_bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
                floor = st.number_input("Floor Number", min_value=0, step=1)
                balcony = st.checkbox("Balcony")
                parking = st.checkbox("Parking")
                uploaded_images = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
            
                if submitted and not uploaded_images:
                    st.warning("At least one image is required.", icon="⚠️")
            
                # Re-render the actual visible button at the bottom
                colx1, colx2, colx3 = st.columns([1, 0.5, 1])
                with colx2:
                    st.form_submit_button("Submit Entry", type="primary")
            
                # Handle logic after form validation
                if submitted and (
                    title.strip()
                    and short_description.strip()
                    and location.strip()
                    and price > 0
                    and size > 0
                    and uploaded_images
                ):
                    all_features = []
                    for image_file in uploaded_images:
                        st.info(f"📤 Analyzing {image_file.name}")
                        items = extract_features(image_file, palm)
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
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust ratios as needed
        with col2:
            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            if st.button("Finalize Entry", use_container_width=True):
                inputs = st.session_state.form_inputs
            
                # Generate final description
                combined_text = generate_combined_text(
                    inputs["title"], inputs["short_description"], inputs["location"],
                    inputs["price"], inputs["size"], inputs["num_bedrooms"],
                    inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
                    detected_features=confirmed_features
                )
            
                embedding = model.encode(combined_text)
                property_uuid = str(uuid.uuid4())
            
                # Upload each image to Hugging Face and collect URLs
                image_urls = [upload_image_to_hub(img, property_uuid) for img in inputs["uploaded_images"]]
            
                entry = create_property_entry(
                    property_uuid,
                    inputs["title"], inputs["short_description"], inputs["location"],
                    inputs["price"], inputs["size"], inputs["num_bedrooms"],
                    inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
                    confirmed_features, embedding, image_urls
                )
                entry["combined_text"] = combined_text
            
                # Save the entry and upload updated DB
                st.session_state.entries.append(entry)
                upload_json_to_hub(st.session_state.entries)
            
                # Reset session state
                st.session_state.pending_features = None
                st.session_state.form_inputs = {}
            
                st.success("✅ Entry saved.")
                st.json(entry)

elif mode == "🔎 Search Properties":
    st.title("🔎 Search Properties")

    query = st.text_input("Describe what you're looking for:")
    if st.button("Search"):
        if query.strip():
            query_embedding = model.encode(query)

            # Do similarity search against loaded entries
            results = []
            for entry in st.session_state.entries:
                entry_embedding = np.array(entry["embedding"])
                score = np.dot(query_embedding, entry_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
                )
                results.append((score, entry))

            results.sort(reverse=True, key=lambda x: x[0])
            top_results = results[:5]

            for score, entry in top_results:
                st.markdown(f"### 📌 {entry['title']} (Score: {score:.2f})")
                st.markdown(entry["short_description"])
                if entry["image_paths"]:
                    st.image(entry["image_paths"][0], width=300)
                st.markdown("---")
        else:
            st.warning("Please enter a search query.")



