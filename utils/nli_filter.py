# utils/nli_filter.py

import torch
import torch.nn.functional as F
import streamlit as st
from sentence_transformers import CrossEncoder

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")

def nli_contradiction_filter(query, results, model, contradiction_threshold=0.2):
    from sentence_transformers.util import batch_to_device

    premises = [query] * len(results)
    hypotheses = [r['data']['combined_text'] for r in results]

    # Prepare inputs
    model_inputs = model.tokenizer(premises, hypotheses, padding=True, truncation=True, return_tensors="pt")
    model_inputs = batch_to_device(model_inputs, model.model.device)

    with torch.no_grad():
        logits = model.model(**model_inputs).logits

    # Convert logits to probabilities using softmax
    probs = F.softmax(logits, dim=1).cpu().numpy()

    filtered = []
    for i, r in enumerate(results):
        contradiction_prob = probs[i][0]
        entailment_prob = probs[i][2]

        r['nli_scores'] = {
            'contradiction': float(contradiction_prob),
            'neutral': float(probs[i][1]),
            'entailment': float(entailment_prob)
        }
        r['nli_score'] = float(entailment_prob)

        if contradiction_prob < contradiction_threshold:
            filtered.append(r)

    return filtered
