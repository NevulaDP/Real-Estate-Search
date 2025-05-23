# utils/nli_filter.py

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")

def split_query(query):
    # Split on punctuation or "and" (expandable)
    parts = query.replace(" and ", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

def nli_contradiction_filter(query, results, model, contradiction_threshold=0.01, entailment_threshold=0.65):
    from sentence_transformers.util import batch_to_device

    sub_queries = split_query(query)

    filtered = []
    for r in results:
        all_entailments = []
        all_contradictions = []

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
            all_entailments.append(probs[2])      # entailment
            all_contradictions.append(probs[0])   # contradiction

        avg_entailment = sum(all_entailments) / len(all_entailments)
        max_contradiction = max(all_contradictions)

        r['nli_scores'] = {
            'contradiction': float(max_contradiction),
            'entailment': float(avg_entailment)
        }
        r['nli_score'] = float(avg_entailment)

        if max_contradiction < contradiction_threshold and avg_entailment >= entailment_threshold:
            filtered.append(r)

    return filtered
