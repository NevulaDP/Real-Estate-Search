# utils/nli_filter.py

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from typing import List

def load_nli_model():
    return CrossEncoder("cross-encoder/nli-deberta-v3-large")

def split_query(query):
    # Split on punctuation or "and" (expandable)
    parts = query.replace(" and ", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

def nli_contradiction_filter(query, results, model, contradiction_threshold=0.01, entailment_threshold=0.6):
    from sentence_transformers.util import batch_to_device

    sub_queries = split_query(query)

    filtered = []
    for r in results:
        all_entailments = []
        all_contradictions = []

        for sub_q in sub_queries:
            inputs = model.tokenizer(
                sub_q,
                r['data']['short_text'],
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
    
def filter_by_entailment_gap(results: List[dict], top_n: int = 3, margin: float = 0.02, min_threshold: float = 0.90) -> List[dict]:
    """
    Filters based on a dynamic entailment gap between entailment and contradiction.
    """
    #if len(results) < 2:
        #return results  # Avoid division by zero on small sets
    for r in results:
        entail = r['nli_scores'].get("entailment", 0)
        contra = r['nli_scores'].get("contradiction", 0)
        r["entailment_gap"] = entail - contra

    sorted_by_gap = sorted(results, key=lambda r: r["entailment_gap"], reverse=True)
    top_gaps = [r["entailment_gap"] for r in sorted_by_gap[:top_n]]

    threshold = max(sum(top_gaps) / len(top_gaps) - margin, min_threshold)

    return [r for r in sorted_by_gap if r["entailment_gap"] >= threshold]
