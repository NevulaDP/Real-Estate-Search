"""
search_section.py

Handles semantic property search in the Property Matcher app.

Flow:
1. Accepts a natural language query from the user.
2. Sends it to Gemini (via `rewrite_query_with_constraints`) to refine the query.
3. Extracts structured constraints from the rewritten query (e.g. price, size).
4. Applies constraint filtering and builds a FAISS index for semantic search.
5. Ranks results by semantic similarity using cosine distance.
6. Uses FLAN (via `verify_claim_flan`) to filter contradiction cases.
7. Displays verified, ranked property results to the user.
"""

import streamlit as st
import numpy as np
import torch
import gc
from sklearn.metrics.pairwise import cosine_similarity


from utils.query_rewrite import rewrite_query_with_constraints
from utils.constraint_filter import (
    extract_constraints_from_query,
    apply_constraint_filters,
    filter_semantic_subqueries
)
from utils.search_embeddings import (
    load_embedding_model,
    build_faiss_index,
    encode_query,
    query_index
)
from utils.inferring_filter import (
    load_verification_model,
    flan_filter,
    verify_claims_batch_flan
)
from utils.hf_loader import load_entries_from_hub

from utils.evals_funcs import (
    log_results_to_csv,
    log_semantic_false_negatives,
    log_faiss_false_negatives
)


def render_search(model, dev_mode=False):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔎 Smart Property Search</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Find the perfect property — just describe what you're looking for</h4>", unsafe_allow_html=True)
        st.markdown("---")
        user_query = st.text_input("What are you looking for in a property?", placeholder="e.g., modern apartment in Tel Aviv with balcony, under $2M")

        if not user_query:
            return
     
         # Step 1: Rewrite query using Gemini
        status = st.empty()
        rewritten = rewrite_query_with_constraints(user_query) # Gemini-powered
        status.empty()
        
        # Show rewritten query
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
        
        # Step 2: Load property data
        status.info("📦 Loading property data...")
        try:
            data = load_entries_from_hub()
        except:
            st.error("Failed to load data.")
            return
        
        # Step 3: Apply structured filters
        constraints = extract_constraints_from_query(rewritten)
        constraints_found = any(constraints.values())
        filtered_data = apply_constraint_filters(data, constraints) if constraints_found else data

        if not filtered_data:
            status.warning("No properties match your query. Try simplifying it.")
            return
        
        # Step 4: Build FAISS index
        status.info("🔍 Indexing...")
        embeddings = np.array([d['embedding'] for d in filtered_data]).astype('float32')
        ids = [d['id'] for d in filtered_data]
        index = build_faiss_index(embeddings)
        query_embedding = encode_query(rewritten, model)
        initial_results, rejected_results = query_index(index, query_embedding, filtered_data, ids, k=len(filtered_data), score_threshold=0.45)

        if not initial_results:
            status.warning("No results found after embedding search.")
            return

        for i, r in enumerate(initial_results):
            r['data']['faiss_score'] = round(r['score'], 3)
            r['data']['faiss_rank'] = i + 1
        
        # Step 5: Semantic ranking using cosine similarity
        sub_queries_to_check = filter_semantic_subqueries(rewritten, constraints) if constraints_found else [rewritten]
        candidates = initial_results

        if sub_queries_to_check:
            semantic_focus = ". ".join(sub_queries_to_check)
            query_vector = model.encode(semantic_focus).reshape(1, -1)
            embedding_matrix = np.array([r['data']['embedding'] for r in candidates])
            similarity_scores = cosine_similarity(query_vector, embedding_matrix)[0]

            del query_vector, embedding_matrix
            gc.collect()

            for i, r in enumerate(candidates):
                r['semantic_similarity'] = float(similarity_scores[i])

            avg_semantic = sum(similarity_scores) / len(similarity_scores)
            threshold = avg_semantic * 0.95
            filtered_candidates = [r for r in candidates if r['semantic_similarity'] >= threshold]
            top_candidates_by_score = sorted(filtered_candidates, key=lambda r: r['semantic_similarity'], reverse=True)
            
            
            top_ids = {r['data']['id'] for r in top_candidates_by_score}
            for r in candidates:
                r['passed_semantic'] = r['data']['id'] in top_ids
        
        # Step 6: FLAN NLI filtering            
        all_candidates = candidates
        skip_flan = not sub_queries_to_check
        recovered_from_rejected = []

        if not skip_flan:
            status.info("🧠 Verifying results")
            flan_model = load_verification_model(force_cpu=True)
            candidates = flan_filter(semantic_focus, top_candidates_by_score, model=flan_model)

            for entry in rejected_results:
                full_text = " ".join([
                    entry['data'].get('short_text', ''),
                    entry['data'].get('description', ''),
                    " ".join(entry['data'].get('features', []))
                ])
                flan_response = verify_claim_flan(flan_model, semantic_focus, full_text)
                entry['flan_response'] = flan_response
                entry['flan_verified'] = flan_response.startswith("true")
                if entry['flan_verified']:
                    recovered_from_rejected.append(entry)
            
            # Developer debug mode: log false negatives
            if dev_mode:
                rejected_candidates = [r for r in all_candidates if not r.get("passed_semantic")]
                flan_dev_results = flan_filter(semantic_focus, rejected_candidates, model=flan_model)
                missed = [r for r in flan_dev_results if r.get("flan_verified")]
                if missed:
                    log_semantic_false_negatives(user_query, missed)

            del flan_model

        for r in all_candidates:
            r['included'] = r in candidates
        if dev_mode:
            log_results_to_csv(rewritten, initial_results)
        if recovered_from_rejected and dev_mode:
            log_faiss_false_negatives(recovered_from_rejected)

        status.empty()

        if not candidates:
            st.warning("No results remain after contradiction filtering.")
            return
        
        # Step 8: Display each matched property
        for i, entry in enumerate(candidates, start=1):
            prop = entry['data']
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

                if prop.get("detected_features"):
                    st.markdown("### 🧠 Included Features")
                    for f in prop["detected_features"]:
                        st.markdown(f"- **{f['item']}**: {f['description']}")

                if prop.get("image_paths"):
                    st.markdown("### 🖼️ Uploaded Images")
                    cols = st.columns(min(3, len(prop["image_paths"])))
                    for i, img_url in enumerate(prop["image_paths"]):
                        with cols[i % len(cols)]:
                            st.image(img_url, use_container_width=True)
        
        # Final cleanup
        del embeddings, index, query_embedding, initial_results, candidates, all_candidates, rejected_results, recovered_from_rejected
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


