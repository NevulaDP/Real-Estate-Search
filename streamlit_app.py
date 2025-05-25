import streamlit as st
from PIL import Image
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
#from utils.form_validation import validate_and_process_form

# Configure Gemini
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

#Monitor

def log_resource_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    cpu = process.cpu_percent(interval=0.1)
    return mem, cpu

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")
    
model = load_embedding_model()

st.set_page_config(page_title="Property Matcher", layout="wide")

# Init session states
if "entries" not in st.session_state:
    st.session_state.entries = load_entries_from_hub()

if "pending_features" not in st.session_state:
    st.session_state.pending_features = None

if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}

if "mode" not in st.session_state:
    st.session_state.mode = "Search"
# --- SIDEBAR ---#
st.sidebar.markdown("""
    <style>
        .sidebar-title {
            font-size: 1.6rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .stButton > button {
            width: 100%;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            background-color: #2e2e2e;
            border: 2px solid #325d7a;
            border-radius: 10px;
            text-align: left;
            transition: 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #008eed;
            border-color: #008eed;
            color: #ddd;
        }
        .stButton.active > button {
            background-color: #008eed !important;
            color: #ddd !important;
            border-color: #008eed;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-title">🏡 Property Matcher</div>', unsafe_allow_html=True)

# Ensure mode is initialized
if "mode" not in st.session_state:
    st.session_state.mode = "Search"

# Create buttons and wrap in containers for styling
upload_btn = st.sidebar.container()
search_btn = st.sidebar.container()

with upload_btn:
    if st.button("Upload Property"):
        st.session_state.mode = "Upload"
with search_btn:
    if st.button("Search Properties"):
        st.session_state.mode = "Search"

# Add 'active' class using JS
st.sidebar.markdown(f"""
    <script>
    const buttons = window.parent.document.querySelectorAll('.stButton');
    if (buttons.length >= 2) {{
        buttons[0].classList.remove("active");
        buttons[1].classList.remove("active");

        {"buttons[0].classList.add('active');" if st.session_state.mode == "Upload" else ""}
        {"buttons[1].classList.add('active');" if st.session_state.mode == "Search" else ""}
    }}
    </script>
""", unsafe_allow_html=True)

mode = st.session_state.mode
#----------------#

if mode == "Upload":
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




elif mode == "Search":
    import json
    import numpy as np
    import streamlit as st
    import torch

    from utils.query_rewrite import rewrite_query_with_constraints
    from utils.constraint_filter import extract_constraints_from_query, apply_constraint_filters, filter_semantic_subqueries
    from utils.search_embeddings import load_embedding_model, build_faiss_index, encode_query, query_index
    from utils.nli_filter import nli_contradiction_filter, load_nli_model, filter_by_entailment_gap
    from utils.hf_loader import load_entries_from_hub
    from sentence_transformers import CrossEncoder
    from sklearn.metrics.pairwise import cosine_similarity
    from utils.lexical_filter import compute_lexical_boost, extract_key_phrases, lexical_entailment_filter, apply_lexical_boost

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔎 Smart Property Search</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Find the perfect property — just describe what you're looking for</h4>", unsafe_allow_html=True)
        st.markdown("---")
        with st.container():
            st.markdown("<div style='margin-top: 0rem;'></div>", unsafe_allow_html=True)
            user_query = st.text_input("What are you looking for in a property?", placeholder="e.g., modern apartment in Tel Aviv with balcony, under $2M")
            #st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
    
        if user_query:
            status = st.empty()
    
            
            rewritten = rewrite_query_with_constraints(user_query)
            status.empty()
            with st.container(border=True):
                st.markdown(f"""
                    <div style='text-align: center; margin-bottom: 0.5rem;'>
                        <span style='color: #549ff0; font-weight: 800; font-size: 1.3rem;'>Refined Query</span>
                    </div>
                    <blockquote style='
                        margin: 1 auto;
                        padding: 0.75rem 1.25rem;
                        background-color: rgba(255, 255, 255, 0.03);
                        border-left: 4px solid #549ff0;
                        border-right: 4px solid #549ff0;
                        font-style: italic;
                        text-align: center;
                        color: #549ff0;
                        font-size: 0.95rem;
                    '>
                        {rewritten}
                    </blockquote>
                """, unsafe_allow_html=True)
                #st.markdown(f"> *{rewritten}*", unsafe_allow_html=True)
            st.markdown("<div style='margin-top: 0.75rem'></div>", unsafe_allow_html=True)
            status = st.empty()
            status.info("🔄 Rewriting your query...")
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
            initial_results = query_index(index, query_embedding, filtered_data, ids, k=20, score_threshold=0.0)
    
            if not initial_results:
                status.empty()
                st.warning("No results found after embedding search.")
                st.stop()
    
            status.info("📊 Reranking results...")
            cross_model = CrossEncoder("cross-encoder/nli-deberta-v3-large")
            pairs = [(f"Required features: {rewritten}", r['data'].get('semantic_text', r['data']['short_text'])) for r in initial_results]

            cross_scores = cross_model.predict(pairs)
    
    
            for i, r in enumerate(initial_results):
                r['rerank_score'] = float(cross_scores[i][2])
    
            reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
            
            
            sub_queries_to_check = []

            if not constraints_found:
                # No constraints? Use full rewritten query
                sub_queries_to_check = [rewritten]
            else:
                # Use only the non-quantified subqueries
                sub_queries_to_check = filter_semantic_subqueries(rewritten, constraints)
            reranked_boosted = reranked
            if sub_queries_to_check:
                status.info("🔎 Running semantic similarity filter...")

                semantic_focus = ". ".join(sub_queries_to_check)
                query_vector = embedding_model.encode(semantic_focus).reshape(1, -1)
                embedding_matrix = np.array([r['data']['embedding'] for r in reranked_boosted])
                similarity_scores = cosine_similarity(query_vector, embedding_matrix)[0]

                del query_vector, embedding_matrix
                gc.collect()

                for i, r in enumerate(reranked_boosted):
                    r['semantic_similarity'] = float(similarity_scores[i])
                    
                reranked = apply_lexical_boost(semantic_focus,reranked,boost_per_hit=0.15);

                #similarity_threshold = 0.91
                
                #--- Dynamic Symantic Treshold
                # After computing semantic_similarity
                similarity_scores = [r['semantic_similarity'] for r in reranked]
                top_k = min(3, len(similarity_scores))  # Look at the top 3 scores
                top_avg = sum(sorted(similarity_scores, reverse=True)[:top_k]) / top_k

                # Dynamic threshold: 85% of the average of top-k
                dynamic_threshold = top_avg * 0.936
                # dynamic_threshold = max(0.3, min(dynamic_threshold, 1)) - forgiving
                dynamic_threshold = max(0.3, dynamic_threshold)
                filtered_semantic = [r for r in reranked if r['semantic_similarity'] >= dynamic_threshold]
                st.caption(f"📊 Dynamic threshold set to **{dynamic_threshold:.3f}** based on top-{top_k} average.")

                if not filtered_semantic:
                    st.warning("✨ We didn’t find a perfect match, but here are the most relevant properties we found.")
                    reranked_boosted=reranked

                if filtered_semantic:
                    reranked_boosted = sorted(filtered_semantic, key=lambda r: r['semantic_similarity'], reverse=True)

                with st.expander("🧠 Semantic Similarity Debug"):
                    for r in reranked_boosted:
                        st.write(f"🏡 {r['data']['title']} → Similarity: {r['semantic_similarity']:.3f}")
    
                

            #########
    
            status.info("🧠 Filtering contradictions...")
            #nli_tokenizer, nli_model = load_nli_model()
            nli_model = load_nli_model()
            filtered_results = nli_contradiction_filter(rewritten, reranked_boosted, model=nli_model, contradiction_threshold=0.015)
            #--- Dynamic Gap Treshold
            # for r in filtered_results:
                # r["entailment_gap"] = r['nli_scores']['entailment'] - r['nli_scores']['contradiction']

            # # Sort by gap
            # sorted_by_gap = sorted(filtered_results, key=lambda r: r["entailment_gap"], reverse=True)

            # # Use top 3 as reference
            # top_n = 4
            # top_gaps = [r["entailment_gap"] for r in sorted_by_gap[:top_n]]
            # dynamic_gap = max(min(top_gaps), 0.9)  # prevent it from being too low
            # filtered_results = [
                # r for r in filtered_results
                # if r['nli_scores']['entailment'] - r['nli_scores']['contradiction'] > dynamic_gap
            # ]
            filtered_results = filter_by_entailment_gap(filtered_results, top_n=3, margin=0.02, min_threshold=0.90)
            for r in filtered_results:
                st.write(f" {r['data']['title']} :{r['entailment_gap']}")
            filtered_results = sorted(filtered_results, key=lambda x: x['rerank_score'], reverse=True)
            ##########
            
            with st.expander("🧪 NLI Debug Output"):
                st.write("Query:", rewritten)
                for r in filtered_results:
                    scores = r.get("nli_scores", {})
                    st.write(f"🧠 {r['data']['title']}")
                    st.write(f"- Contradiction: {scores.get('contradiction', 0):.3f}")
                    st.write(f"- Entailment: {scores.get('entailment', 0):.3f}")
            
            ##########
    
            status.empty()  # Clear the loading messages
    
            if not filtered_results:
                st.warning("No results remain after contradiction filtering.")
                st.stop()
            
           
            for i, entry in enumerate(filtered_results, start=1):
                prop = entry['data']  # ← this line is essential
    
                with st.container(border=True):
                    st.markdown(f"### {i}. {prop['title']}")
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
                        st.markdown("### 🧠 Included Features")
                        for f in prop["detected_features"]:
                            st.markdown(f"- **{f['item']}**: {f['description']}")
    
                    if prop["image_paths"]:
                        st.markdown("### 🖼️ Uploaded Images")
                        cols = st.columns(min(3, len(prop["image_paths"])))
                        for i, img_url in enumerate(prop["image_paths"]):
                            with cols[i % len(cols)]:
                                st.image(img_url, use_container_width=True)
           
            
            # Force garbage collection
            del embeddings, index, query_embedding, initial_results, reranked             
            del cross_model
            del pairs, cross_scores
            del nli_model
            gc.collect()
            
            # Clear GPU cache if you're using a model on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mem, cpu = log_resource_usage()
            st.write(f"📊 Memory usage: {mem:.2f} MB")
            st.write(f"⚙️ CPU usage: {cpu:.2f}%")
