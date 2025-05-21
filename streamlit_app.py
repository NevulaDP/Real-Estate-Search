import streamlit as st
from PIL import Image
import uuid
import google.generativeai as palm
from sentence_transformers import SentenceTransformer

from utils.features import extract_features, generate_combined_text, generate_short_text
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

                embedding = model.encode(short_text)
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

                st.session_state.entries.append(entry)
                upload_json_to_hub(entry)

                st.session_state.pending_features = None
                st.session_state.form_inputs = {}
                st.session_state.upload_stage = "done"
                st.session_state.finalized_entry = entry
                st.rerun()




elif mode == "🔎 Search Properties":
    import json
    import numpy as np
    import streamlit as st

    from utils.query_rewrite import rewrite_query_with_constraints
    from utils.constraint_filter import extract_constraints_from_query, apply_constraint_filters
    from utils.search_embeddings import load_embedding_model, build_faiss_index, encode_query, query_index
    from utils.nli_filter import nli_contradiction_filter, load_nli_model
    from utils.hf_loader import load_entries_from_hub
    from sentence_transformers import CrossEncoder
    from sklearn.metrics.pairwise import cosine_similarity

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔎 Smart Property Search</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Find the perfect property — just describe what you're looking for</h4>", unsafe_allow_html=True)
        with st.container():
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            user_query = st.text_input("What are you looking for in a property?", placeholder="e.g., modern apartment in Tel Aviv with balcony, under $2M")
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
        if user_query:
            st.markdown("---")
            status = st.empty()
    
            status.info("🔄 Rewriting your query...")
            rewritten = rewrite_query_with_constraints(user_query)
            status.empty()
    
            st.markdown(f"> *{rewritten}*", unsafe_allow_html=True)

    
            status.info("📦 Loading property data...")
            try:
                data = load_entries_from_hub()
            except:
                st.error("Failed to load data.")
                st.stop()
    
            constraints = extract_constraints_from_query(rewritten)
            constraints_found = any(constraints.values())
            if not any(constraints.values()):
                filtered_data = data
            else:
                filtered_data = apply_constraint_filters(data, constraints)
    
            if not filtered_data:
                status.empty()
                st.warning("No properties match your query. Try simplifying it.")
                st.stop()
    
            status.info("🔍 Searching...")
            embedding_model = load_embedding_model()
            embeddings = np.array([d['embedding'] for d in filtered_data]).astype('float32')
            #########
    
            ids = [d['id'] for d in filtered_data]
            index = build_faiss_index(embeddings)
    
            query_embedding = encode_query(
                f"{rewritten}",
                embedding_model
            )
            initial_results = query_index(index, query_embedding, filtered_data, ids, k=10, score_threshold=0.0)
    
            if not initial_results:
                status.empty()
                st.warning("No results found after embedding search.")
                st.stop()
    
            status.info("📊 Reranking results...")
            cross_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
            pairs = [(f"Required features: {rewritten}", r['data']['combined_text']) for r in initial_results]
            cross_scores = cross_model.predict(pairs)
    
    
            for i, r in enumerate(initial_results):
                r['rerank_score'] = float(cross_scores[i][2])
    
            reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
            if not constraints_found:
                #########
                # 🔎 Apply semantic similarity filtering
                query_vector = query_embedding.reshape(1, -1)
                embedding_matrix = np.array([r['data']['embedding'] for r in reranked])
                similarity_scores = cosine_similarity(query_vector, embedding_matrix)[0]
                
                # Attach scores
                for i, r in enumerate(reranked):
                    r['semantic_similarity'] = float(similarity_scores[i])
                
                similarity_threshold = 0.4
                filtered_semantic = [r for r in reranked if r['semantic_similarity'] >= similarity_threshold]
                
                if not filtered_semantic:
                        st.warning("✨ We didn’t find a perfect match, but here are the most relevant properties we found.")
                #DEBUG
                # 🧠 Debug output
                #with st.expander("🧠 Semantic Similarity Debug"):
                #   for r in reranked:
                #       st.write(f"🏡 {r['data']['title']} → Similarity: {r['semantic_similarity']:.3f}")
                  
                
                # Fallback if semantic check failed
                if filtered_semantic:
                    reranked = sorted(filtered_semantic, key=lambda r: r['semantic_similarity'], reverse=True)
                # else keep reranked as-is (fallback)
                # DEBUG
                #with st.expander("🧠 Semantic Similarity Debug"):
                #    for r in reranked:
                #        st.write(f"🏡 {r['data']['title']} → Similarity: {r['semantic_similarity']:.3f}")
            
            else:
                # Skip semantic filtering — use reranked as-is
                pass
    
    
            #########
    
            status.info("🧠 Filtering contradictions...")
            #nli_tokenizer, nli_model = load_nli_model()
            nli_model = load_nli_model()
            filtered_results = nli_contradiction_filter(rewritten, reranked, model=nli_model, contradiction_threshold=0.2)
    
            ##########
            
            #with st.expander("🧪 NLI Debug Output"):
            #    st.write("Query:", rewritten)
            #    for r in reranked:
            #        scores = r.get("nli_scores", {})
            #        st.write(f"🧠 {r['data']['title']}")
            #        st.write(f"- Contradiction: {scores.get('contradiction', 0):.3f}")
            #        st.write(f"- Entailment: {scores.get('entailment', 0):.3f}")
            
            ##########
    
            status.empty()  # Clear the loading messages
    
            if not filtered_results:
                st.warning("No results remain after contradiction filtering.")
                st.stop()
           
            for entry in filtered_results:
                prop = entry['data']  # ← this line is essential
    
                with st.container(border=True):
                    st.markdown(f"### 🏡 {prop['title']}")
                    st.markdown(f"*{prop['short_description']}*")
                    st.markdown(f"📍 **Location:** {prop['location']}")
                    st.markdown(f"💰 **Price:** ${prop['price']:,}")
                    st.markdown(f"🛌 **Bedrooms:** {prop['num_bedrooms']}  |  🛁 **Bathrooms:** {prop['num_bathrooms']}  |  🏢 **Floor:** {prop['floor']}")
                    st.markdown(f"🖐️ **Size:** {prop['size']} sq ft")
    
                    extras = []
                    if prop['balcony']:
                        extras.append("🪟 Balcony")
                    if prop['parking']:
                        extras.append("🔃 Parking")
                    if extras:
                        st.markdown("🔧 **Extras:** " + ", ".join(extras))
    
                    if prop["detected_features"]:
                        st.markdown("### 🧠 Detected Features")
                        for f in prop["detected_features"]:
                            st.markdown(f"- **{f['item']}**: {f['description']}")
    
                    if prop["image_paths"]:
                        st.markdown("### 🖼️ Uploaded Images")
                        cols = st.columns(min(3, len(prop["image_paths"])))
                        for i, img_url in enumerate(prop["image_paths"]):
                            with cols[i % len(cols)]:
                                st.image(img_url, use_container_width=True)
    
                    
