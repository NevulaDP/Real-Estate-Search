# utils/inferring_filter.py

import torch
import torch.nn.functional as F
from sentence_transformers import CrossEncoder
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from difflib import SequenceMatcher
import re

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




def split_query(query):
    # Split on punctuation or "and" (expandable)
    parts = query.replace(" and ", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

    

# Load FLAN-T5-XL model

def load_verification_model(force_cpu=False):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    return {"tokenizer": tokenizer, "model": model, "device": device}

# Use FLAN-T5-XL to verify if property supports the query



# def verify_claim_flan(model, claim: str, context: str) -> str:
    # prompt = f"""
        # You are verifying whether a real estate property description supports a specific user requirement.

        # 🔍 Requirement: "{claim}"
        # 📄 Property Description: "{context}"

        # Your task:
        # 1. Decide if the description fully satisfies the requirement.
        # 2. Respond only with "True" or "False".

        # Answer:
        # """

    # inputs = model["tokenizer"](prompt, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(model["device"])

    # with torch.no_grad():
        # output_ids = model["model"].generate(
            # **inputs,
            # max_new_tokens=5,
            # pad_token_id=model["tokenizer"].pad_token_id,
            # eos_token_id=model["tokenizer"].eos_token_id
        # )

    # return model["tokenizer"].decode(output_ids[0], skip_special_tokens=True).strip().lower()

# def flan_filter(query: str, results: list, model) -> list:
    # filtered = []
    # for r in results:
        # full_text = " ".join([
            # r['data'].get('short_text', ''),
            # r['data'].get('description', ''),
            # " ".join(r['data'].get('features', []))
        # ])

        # flan_response = verify_claim_flan(model, query, full_text)
        # r['flan_response'] = flan_response
        # r['flan_verified'] = flan_response.startswith("true")
        # r['flan_match_score'] = 1.0 if r['flan_verified'] else 0.0

        # if r['flan_verified']:
            # filtered.append(r)

    # return filtered
    
    
def flan_filter(query: str, results: list, model) -> list:
    filtered = []

    contexts = [
        " ".join([
            r['data'].get('short_text', ''),
            r['data'].get('description', ''),
            " ".join(r['data'].get('features', []))
        ]) for r in results
    ]

    claims = [query] * len(results)
    responses = verify_claims_batch_flan(model, claims, contexts, batch_size=6)

    for r, response in zip(results, responses):
        r['flan_response'] = response
        r['flan_verified'] = response.startswith("true")
        r['flan_match_score'] = 1.0 if r['flan_verified'] else 0.0
        if r['flan_verified']:
            filtered.append(r)

    return filtered
    

def verify_claims_batch_flan(model, claims: List[str], contexts: List[str], batch_size:6) -> List[str]:
    prompts = [
        f"""
        You are verifying whether a real estate property description supports a specific user requirement.

        🔍 Requirement: "{claim}"
        📄 Property Description: "{context}"

        Your task:
        1. Decide if the description fully satisfies the requirement.
        2. Respond only with "True" or "False".

        Answer:
        """
        for claim, context in zip(claims, contexts)
    ]

    tokenizer = model["tokenizer"]
    device = model["device"]
    model = model["model"]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return [r.strip().lower() for r in decoded]



