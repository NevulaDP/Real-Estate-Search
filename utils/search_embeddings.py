"""
search_embeddings.py

Provides helper functions for embedding-based property search:
- load_embedding_model: loads & caches the SentenceTransformer
- build_faiss_index: constructs a normalized FAISS IndexFlatIP
- encode_query: prepends a real-estate prompt and returns a float32 vector
- query_index: retrieves top-k matches (and “rejected” ones) given a FAISS index

Each function is documented below for usage details.
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedding_model():

    """
    Loads the SentenceTransformer model for embedding queries and documents.
    Cached by Streamlit to avoid reloading on every run.

    Returns:
        SentenceTransformer: Loaded embedding model.
    """

    return SentenceTransformer("BAAI/bge-small-en-v1.5")

@st.cache_resource
def build_faiss_index(embeddings):

    """
    Builds a FAISS index using inner product (dot product) similarity on normalized vectors.

    Args:
        embeddings (np.ndarray): 2D numpy array of shape (n_samples, embedding_dim).
    
    Returns:
        faiss.IndexFlatIP: A FAISS index for fast nearest neighbor search.
    """

    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def encode_query(query, model):

    """
    Encodes a user query into an embedding using the provided model.
    
    Args:
        query (str): Raw user input query.
        model: Loaded SentenceTransformer model.
    
    Returns:
        np.ndarray: A single embedding vector for the query.
    """

    prefix = "Represent this real estate query for retrieving relevant property listings: "
    full_query = prefix + query
    embedding = model.encode(full_query).astype('float32')
    return embedding


def query_index(index, query_embedding, metadata, ids, k=5, score_threshold=0.45):

    """
    Performs a similarity search on the FAISS index and returns matching metadata.

    Args:
        index (faiss.Index): FAISS search index.
        query_embedding (np.ndarray): Embedding vector of the query.
        metadata (list): List of metadata dictionaries (each must include 'id').
        ids (list): List of IDs corresponding to indexed items.
        k (int, optional): Number of nearest neighbors to retrieve. Default is 5.
        score_threshold (float, optional): Minimum similarity score to consider a match.
    
    Returns:
        tuple:
            results (list): Entries with scores above threshold.
            rejected (list): Entries below threshold.
    """

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
