# utils/inferring_filter.py

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig

SYNONYM_MAP = {
    "tv": "television",
    "tele": "television",
    "flat screen": "television",
    "fridge": "refrigerator",
    "oven": "stove",
    "wardrobe": "closet",
    "sofa": "couch",
    "lavatory": "bathroom",
    "wc": "bathroom",
}


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
    if not top_gaps:
        return [] 
    threshold = max(sum(top_gaps) / len(top_gaps) - margin, min_threshold)

    return [r for r in sorted_by_gap if r["entailment_gap"] >= threshold]

# Load FLAN-T5-XL model

def load_verification_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return {"tokenizer": tokenizer, "model": model, "device": device}

# Use FLAN-T5-XL to verify if property supports the query

def verify_claim_flan(model, claim: str, context: str) -> str:
    prompt = f"""
                You are verifying whether a property description supports a specific user requirement.

                🔍 Claim: "{claim}"
                📄 Property Description: "{context}"

                Your task:
                1. Determine if the property description fully satisfies the claim.
                2. Output only "True" or "False" as your answer.
                3. Provide a one-line factual justification.
                4. Briefly explain your reasoning in 1–2 sentences.

                📝 Respond in the following format exactly:

                Answer: <True or False>
                Justification: <short sentence>
                Explanation: <1-2 sentence explanation>

                Answer: True
                Justification:"""

    inputs = model["tokenizer"](prompt, return_tensors="pt", truncation=True).to(model["device"])

    with torch.no_grad():
        output_ids = model["model"].generate(**inputs, max_new_tokens=100)

    return model["tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip().lower()
    #return response.startswith("yes")

# Run verification over all candidate properties


def flan_filter(query: str, results: list, model) -> list:
    filtered = []
    for r in results:
        full_text = " ".join([
            r['data'].get('short_text', ''),
            r['data'].get('description', ''),
            " ".join(r['data'].get('features', []))
        ])

        flan_response = verify_claim_flan(model, query, full_text)
        r['flan_response'] = flan_response
        r['flan_verified'] = flan_response.lower().startswith("true")  # match prompt

        r['flan_match_score'] = score_flan_match_quality(
            flan_response,
            constraint=query,
            detected_features=r['data'].get('detected_features', [])
        )

        if r['flan_verified']:
            filtered.append(r)

    return filtered

def score_flan_match_quality(response_text: str, constraint: str, detected_features: list) -> float:
    """
    Scores the Flan verification response:
    - 1.0 = clear 'True' with constraint + feature
    - 0.75 = 'True' and includes constraint or feature
    - 0.5 = 'True' but vague or generic
    - 0.25 = 'False' but has some relevant justification
    - 0.0 = irrelevant, unclear, or wrong
    """
    response = normalize_text(response_text.strip())
    constraint = normalize_text(constraint)

    has_true = response.startswith("true")
    has_false = response.startswith("false")

    mentions_constraint = constraint in response

    mentions_feature = any(
        normalize_text(feature.get("item", "")) in response or
        normalize_text(feature.get("description", "")) in response
        for feature in detected_features
    )

    if has_true and mentions_constraint and mentions_feature:
        return 1.0
    elif has_true and (mentions_constraint or mentions_feature):
        return 0.75
    elif has_true:
        return 0.5
    elif has_false and (mentions_constraint or mentions_feature):
        return 0.25
    else:
        return 0.0




def normalize_text(text: str) -> str:
    text = text.lower()
    for synonym, canonical in SYNONYM_MAP.items():
        text = text.replace(synonym, canonical)
    return text