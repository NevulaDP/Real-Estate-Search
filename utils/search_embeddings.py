# utils/search_embeddings.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def encode_query(query, model):
    embedding = model.encode(query).astype('float32')
    return embedding / np.linalg.norm(embedding)

def query_index(index, query_embedding, metadata, ids, k=5, score_threshold=0.45):
    if index.ntotal == 0:
        return []

    query_embedding = np.array([query_embedding])
    scores, indices = index.search(query_embedding, k)
    results = []

    for i in range(k):
        score = scores[0][i]
        if indices[0][i] != -1 and score >= score_threshold:
            result_id = ids[indices[0][i]]
            for item in metadata:
                if item['id'] == result_id:
                    results.append({
                        'score': float(score),
                        'data': item
                    })
                    break

    return results
