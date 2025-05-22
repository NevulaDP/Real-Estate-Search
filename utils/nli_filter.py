# utils/nli_filter.py

import torch
import torch.nn.functional as F
import streamlit as st
from sentence_transformers import CrossEncoder

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")

def split_query(query):
    # Split on period or "and" (basic version, can improve)
    parts = query.replace(" and ", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

def nli_contradiction_filter(query, results, model, contradiction_threshold=0.2):
    from sentence_transformers.util import batch_to_device

    sub_queries = split_query(query)

    for r in results:
        entailment_scores = []
        contradiction_flags = []

        for sub_q in sub_queries:
            inputs = model.tokenizer(
                sub_q,
                r['data']['combined_text'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            inputs = inputs.to(model.model.device)

            with torch.no_grad():
                logits = model.model(**inputs).logits

            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            entailment_scores.append(probs[2])  # entailment
            contradiction_flags.append(probs[0] < contradiction_threshold)

        r['nli_scores'] = {
            'contradiction': 1.0 - min(contradiction_flags),  # 0 if any contradict, else 1
            'entailment': sum(entailment_scores) / len(entailment_scores)
        }
        r['nli_score'] = r['nli_scores']['entailment']

    # Only return if *none* of the subqueries contradict
    return [r for r in results if all(r['nli_scores']['contradiction'])]
