"""
inferring_filter.py

Provides functions for verifying whether a property listing satisfies a user's query
using the FLAN-T5-XL model. Supports both batched and individual claim evaluation.

Key Functions:
- load_verification_model: loads and returns a FLAN-T5-XL model with tokenizer and device config.
- flan_filter: filters a list of search results by verifying each against the user's query.
- verify_claims_batch_flan: performs batched FLAN verification for efficiency.
"""

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def split_query(query):

    """
    Splits a query into individual sub-queries based on punctuation or 'and'.

    Args:
        query (str): Raw user input string.

    Returns:
        List[str]: Cleaned list of query segments.
    """

    parts = query.replace(" and ", ".").split(".")
    return [p.strip() for p in parts if p.strip()]

    

def load_verification_model(force_cpu=False):

    """
    Loads the FLAN-T5-XL model and tokenizer, automatically selecting the device.

    Args:
        force_cpu (bool): If True, forces the model to run on CPU even if GPU is available.

    Returns:
        dict: Dictionary containing tokenizer, model, and device.
    """

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model = model.to(device)
    return {"tokenizer": tokenizer, "model": model, "device": device}
    
    
def flan_filter(query: str, results: list, model) -> list:

    """
    Filters a list of property results by verifying each listing against the user query
    using the FLAN model in batch mode.

    Args:
        query (str): The user’s query string.
        results (list): List of property results (each with 'short_text', 'description', 'features').
        model (dict): Dictionary with FLAN model, tokenizer, and device.

    Returns:
        list: Filtered results that passed FLAN verification.
    """

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

    """
    Performs batch claim verification using FLAN-T5-XL.

    Args:
        model (dict): Dictionary with FLAN model, tokenizer, and device.
        claims (List[str]): List of user queries.
        contexts (List[str]): Corresponding property descriptions.
        batch_size (int): Number of claims to process per batch, Controls memory/performance tradeoff.

    Returns:
        List[str]: FLAN responses ("true" or "false") per input.
    """

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

    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_responses.extend([r.strip().lower() for r in decoded])

    return all_responses



