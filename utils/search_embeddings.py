# utils/search_embeddings.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def encode_query(query, model):
    prefix = "Represent this real estate query for retrieving relevant property listings: "
    full_query = prefix + query
    embedding = model.encode(full_query).astype('float32')
    return embedding


def query_index(index, query_embedding, metadata, ids, k=5, score_threshold=0.45):
    if index.ntotal == 0:
        return []

    query_embedding = np.array([query_embedding])
    scores, indices = index.search(query_embedding, k)
    results = []
    rejected= []

    for i in range(k):
        score = scores[0][i]
        if indices[0][i] != -1:
            result_id = ids[indices[0][i]]
            for item in metadata:
                if item['id'] == result_id:
                    entry ={
                        'score': float(score),
                        'data': item
                    }
                    if score >= score_threshold:
                        results.append(entry)
                    else:
                        rejected.append(entry)
                    break
    return results, rejected
