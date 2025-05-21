# utils/nli_filter.py

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource
def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/nli-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-roberta-base")
    return tokenizer, model

def get_nli_scores(query, text, tokenizer, model):
    inputs = tokenizer.encode_plus(query, text, return_tensors='pt', truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return {
        "contradiction": probs[0].item(),
        "neutral": probs[1].item(),
        "entailment": probs[2].item()
    }

def nli_contradiction_filter(query, results, tokenizer, model, contradiction_threshold=0.2):
    filtered = []
    for r in results:
        text = r['data']['combined_text']
        scores = get_nli_scores(query, text, tokenizer, model)

        if scores['contradiction'] < contradiction_threshold:
            r['nli_scores'] = scores
            r['nli_score'] = scores['entailment']
            filtered.append(r)
        else:
            print(f"❌ CONTRADICTION: '{r['data']['title']}'")

        print(f"🏠 {r['data']['title']} → Contradiction: {scores['contradiction']:.3f} | Entailment: {scores['entailment']:.3f}")
    return filtered
