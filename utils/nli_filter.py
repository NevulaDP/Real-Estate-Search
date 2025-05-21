# utils/nli_filter.py

import torch
import streamlit as st
from sentence_transformers import CrossEncoder

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")

def nli_contradiction_filter(query, results, model=None, contradiction_threshold=0.01):
    """
    Filters out results that contradict the user query using entailment score.
    Assumes model is a sentence-transformers CrossEncoder that outputs 3-way NLI scores.
    """
    if model is None:
        model = load_nli_model()

    pairs = [(query, r["data"]["combined_text"]) for r in results]
    scores = model.predict(pairs)

    filtered = []
    for r, logits in zip(results, scores):
        contradiction_prob = logits[0]  # index 0 = contradiction
        entailment_prob = logits[2]     # index 2 = entailment

        r["nli_score"] = entailment_prob
        if contradiction_prob < contradiction_threshold:
            filtered.append(r)
        else:
            print(f"❌ CONTRADICTION: {r['data']['title']} | Contradiction: {contradiction_prob:.2f}")

 

    
    return filtered

