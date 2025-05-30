import streamlit as st
import uuid
import google.generativeai as palm
from sentence_transformers import SentenceTransformer
import os # Monitor
import psutil # Monitor
import gc


from utils.features import extract_features, generate_combined_text, generate_short_text, generate_semantic_text
from utils.database import create_property_entry
from utils.hf_uploader import upload_image_to_hub, upload_json_to_hub
from utils.hf_loader import load_entries_from_hub

def render_upload(model, dev_mode = False):
        # --- Handle upload stage logic ---
    if "upload_stage" not in st.session_state:
       st.session_state.upload_stage = "form"

    if st.session_state.upload_stage == "done":
        entry = st.session_state.pop("finalized_entry")
        st.session_state.upload_stage = "form"  # Reset for next round

        st.success("✅ Property submitted successfully!")

        # Center the confirmation inside a full-width container
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.container():
                st.markdown(f"### 🏡 {entry['title']}")
                st.markdown(f"*{entry['short_description']}*")
                st.markdown(f"📍 **Location:** {entry['location']}")
                st.markdown(f"💰 **Price:** ${entry['price']:,}")
                st.markdown(f"🛏️ **Bedrooms:** {entry['num_bedrooms']}  |  🛁 **Bathrooms:** {entry['num_bathrooms']}  |  🏢 **Floor:** {entry['floor']}")
                st.markdown(f"📐 **Size:** {entry['size']} sq ft")

                extras = []
                if entry['balcony']:
                    extras.append("🪟 Balcony")
                if entry['parking']:
                    extras.append("🅿️ Parking")
                if extras:
                    st.markdown("🔧 **Extras:** " + ", ".join(extras))

                if entry["detected_features"]:
                    st.markdown("### 🧠 Detected Features")
                    for f in entry["detected_features"]:
                        st.markdown(f"- **{f['item']}**: {f['description']}")

                if entry["image_paths"]:
                    st.markdown("### 🖼️ Uploaded Images")
                    cols = st.columns(min(3, len(entry["image_paths"])))
                    for i, img_url in enumerate(entry["image_paths"]):
                        with cols[i % len(cols)]:
                            st.image(img_url, use_container_width=True)

        st.stop()

    # --- Phase 1: Form submission ---
    if st.session_state.upload_stage == "form":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.title("🏡 Upload Property")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.form("property_form"):
                title = st.text_input("Title *")
                short_description = st.text_area("Short Description *")
                location = st.text_input("Location *")
                price = st.number_input("Price ($)", min_value=1, step=1000)
                size = st.number_input("Size (sq ft)", min_value=1, step=10)
                num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
                num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)
                floor = st.number_input("Floor Number", min_value=0, step=1)
                balcony = st.checkbox("Balcony")
                parking = st.checkbox("Parking")
                uploaded_images = st.file_uploader("Upload Image(s) *", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

                colx1, colx2, colx3 = st.columns([1, 0.5, 1])
                with colx2:
                    submitted = st.form_submit_button("Submit Entry")

                if submitted and not title.strip():
                    st.warning("Title is required.", icon="⚠️")
                if submitted and not short_description.strip():
                    st.warning("Short description is required.", icon="⚠️")
                if submitted and not location.strip():
                    st.warning("Location is required.", icon="⚠️")
                if submitted and price <= 0:
                    st.warning("Price must be greater than 0.", icon="⚠️")
                if submitted and size <= 0:
                    st.warning("Size must be greater than 0.", icon="⚠️")
                if submitted and not uploaded_images:
                    st.warning("At least one image is required.", icon="⚠️")

                if submitted and (
                    title.strip() and short_description.strip() and location.strip() and
                    price > 0 and size > 0 and uploaded_images
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
                    st.session_state.upload_stage = "features"
                    st.rerun()

    # --- Phase 2: Confirm features and finalize ---
    if st.session_state.upload_stage == "features":
        confirmed_features = []
        features = st.session_state.pending_features
        uploaded_images = st.session_state.form_inputs.get("uploaded_images", [])

        num_images = len(uploaded_images)
        features_per_image = len(features) // num_images if num_images else len(features)

        for img_idx, image_file in enumerate(uploaded_images):
            start = img_idx * features_per_image
            end = (img_idx + 1) * features_per_image if img_idx < num_images - 1 else len(features)
            image_features = features[start:end]

            with st.container():
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

                st.markdown("---")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            if st.button("Finalize Entry", use_container_width=True):
                inputs = st.session_state.form_inputs

                combined_text = generate_combined_text(
                    inputs["title"], inputs["short_description"], inputs["location"],
                    inputs["price"], inputs["size"], inputs["num_bedrooms"],
                    inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
                    detected_features=confirmed_features
                )

                short_text = generate_short_text(
                    inputs["title"], inputs["short_description"], inputs["location"],
                    inputs["price"], inputs["size"], inputs["num_bedrooms"],
                    inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
                    confirmed_features
                )
                #semantic text for embeddings creation
                semantic_text = generate_semantic_text(
                inputs["title"], inputs["short_description"], inputs["location"],
                detected_features=confirmed_features
)
                embedding = model.encode(combined_text)
                property_uuid = str(uuid.uuid4())

                image_urls = [upload_image_to_hub(img, property_uuid) for img in inputs["uploaded_images"]]

                entry = create_property_entry(
                    property_uuid,
                    inputs["title"], inputs["short_description"], inputs["location"],
                    inputs["price"], inputs["size"], inputs["num_bedrooms"],
                    inputs["num_bathrooms"], inputs["balcony"], inputs["parking"], inputs["floor"],
                    confirmed_features, embedding, image_urls
                )
                entry["combined_text"] = combined_text
                entry["short_text"] = short_text
                entry["semantic_text"] = semantic_text

                st.session_state.entries.append(entry)
                upload_json_to_hub(entry)

                st.session_state.pending_features = None
                st.session_state.form_inputs = {}
                st.session_state.upload_stage = "done"
                st.session_state.finalized_entry = entry
                st.rerun()
