"""
streamlit_app.py

Main entry point for the Property Matcher Streamlit app.
Handles both Upload and Search modes.

Features:
- Upload real estate listings with AI-analyzed images
- Smart semantic search with FAISS, constraints, and FLAN verification
- Logging, memory monitoring, and developer tools
"""

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
from utils.evals_funcs import log_results_to_csv, attach_keyword_overlap_metrics, extract_keywords, enrich_with_scores, log_semantic_false_negatives,log_faiss_false_negatives
from modules.upload_section import render_upload
from modules.search_section import render_search


#-----Gemini Configuration
palm.configure(api_key=st.secrets["GOOGLE_API_KEY"])

#-----Monitoring

def log_resource_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    cpu = process.cpu_percent(interval=0.1)
    return mem, cpu

#-----Model Cache

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def get_flan_model_cpu():
    model = load_verification_model(force_cpu=True)
    return model
    
    
model = load_embedding_model()


st.set_page_config(page_title="Property Matcher", layout="wide")

#----- Init session states
if "entries" not in st.session_state:
    st.session_state.entries = load_entries_from_hub()

if "pending_features" not in st.session_state:
    st.session_state.pending_features = None

if "form_inputs" not in st.session_state:
    st.session_state.form_inputs = {}

if "mode" not in st.session_state:
    st.session_state.mode = "Search"
#----- SIDEBAR
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

dev_mode = st.sidebar.checkbox("🛠️ Enable Dev Mode")


mode = st.session_state.mode
#----------------#

if mode == "Upload":
    render_upload(model, dev_mode)
    




elif mode == "Search":
    render_search(model, dev_mode)
    
    mem, cpu = log_resource_usage()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write(f"📊 Memory usage: {mem:.2f} MB")
        st.write(f"⚙️ CPU usage: {cpu:.2f}%")