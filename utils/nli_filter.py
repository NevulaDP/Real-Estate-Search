# utils/nli_filter.py

import torch
import streamlit as st
from sentence_transformers import CrossEncoder

def nli_contradiction_filter(query, results, contradiction_threshold=0.2):
    """
    Filters out results that semantically contradict the user's query using a modern cross-encoder.

    Args:
        query (str): The rewritten user query (e.g., "The property must have a kitchen island.")
        results (list): List of dicts with structure {'data': ..., 'score': ..., etc.}
        contradiction_threshold (float): Maximum contradiction score to allow a result through

    Returns:
        list: Filtered and scored results
    """
    model_name = "cross-encoder/nli-deberta-v3-large"
    cross_model = CrossEncoder(model_name)

    # Prepare pairs: (query, property description)
    pairs = [(query, r['data']['combined_text']) for r in results]

    # Predict scores: returns [contradiction, neutral, entailment] for each pair
    raw_scores = cross_model.predict(pairs)

    filtered = []
    for score_vec, result in zip(raw_scores, results):
        contradiction_prob = float(score_vec[0])  # index 0 = contradiction
        entailment_prob = float(score_vec[2])     # index 2 = entailment

        if contradiction_prob < contradiction_threshold:
            result['nli_score'] = entailment_prob
            result['nli_scores'] = {
                'contradiction': contradiction_prob,
                'neutral': float(score_vec[1]),
                'entailment': entailment_prob
            }
            filtered.append(result)
        else:
            print(f"❌ CONTRADICTED: {result['data']['title']}")

    return filtered
