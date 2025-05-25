from rapidfuzz import fuzz
import re
from typing import List
from collections import Counter
import numpy as np

def lexical_entailment_filter(query, results, threshold=60):
    """
    Filters properties based on simple lexical overlap with the query.
    Returns all properties with at least one strong lexical match.
    """
    filtered = []

    for r in results:
        text = r['data'].get("semantic_text") or r['data'].get("short_text", "")
        score = fuzz.partial_ratio(query.lower(), text.lower())

        r["lexical_score"] = score
        if score >= threshold:
            filtered.append(r)

    return filtered
    


SYNONYM_MAP = {
    "close proximity": ["near", "nearby", "close"],
    "sea": ["ocean", "beach", "coast"],
    "kitchen island": ["island", "central counter"]
    # Add more synonym groups as needed
}

def extract_key_phrases(text: str) -> List[str]:
    """
    Naively extracts keyword phrases: looks for adjective+noun or noun+noun patterns.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    phrases = []

    for i in range(len(words) - 1):
        pair = f"{words[i]} {words[i + 1]}"
        if words[i] not in {"the", "a", "an", "is", "are", "and", "or", "in", "of", "with"}:
            phrases.append(pair)

    return list(set(phrases))

def synonym_match(query_phrase: str, semantic_text: str) -> bool:
    """
    Checks if any known synonyms of a query phrase appear in the semantic text.
    """
    for key, synonyms in SYNONYM_MAP.items():
        if key in query_phrase:
            for synonym in synonyms:
                if synonym in semantic_text:
                    return True
    return False

def compute_lexical_boost(query: str, semantic_text: str, boost_per_hit: float = 0.25, max_boost: float = 0.75) -> float:
    """
    Computes a boost score based on direct lexical overlap between query and semantic_text.
    Includes:
    - Exact phrase match
    - Synonym support
    - Partial token overlap
    - Title match boost
    """
    semantic_text = semantic_text.lower()
    phrases = extract_key_phrases(query)

    hits = 0
    for phrase in phrases:
        phrase_cleaned = re.sub(r"[^a-z0-9 ]", "", phrase)
        if phrase_cleaned in semantic_text:
            hits += 1
        elif synonym_match(phrase_cleaned, semantic_text):
            hits += 1
        else:
            # Token overlap fallback (e.g. partial matches like "kitchen island" vs "island")
            phrase_tokens = set(phrase_cleaned.split())
            semantic_tokens = set(re.findall(r"\w+", semantic_text))
            if phrase_tokens & semantic_tokens:
                hits += 0.5  # partial match boost

    # Title bonus: if title (first sentence) contains important words
    title = semantic_text.split(".")[0]
    for word in query.split():
        if word.lower() in title.lower():
            hits += 0.25  # mild bonus for relevant title tokens

    total_boost = min(hits * boost_per_hit, max_boost)
    return total_boost


def apply_lexical_boost(query: str, results: List[dict], boost_per_hit: float = 0.25, max_boost: float = 0.75) -> List[dict]:
    """
    Applies a lexical similarity boost to each result's semantic_similarity score based on keyword overlap.
    """
    for r in results:
        semantic_text = r['data'].get('semantic_text', '')
        boost = compute_lexical_boost(query, semantic_text, boost_per_hit=boost_per_hit, max_boost=max_boost)
        r['semantic_similarity'] += boost
        r['lexical_boost'] = boost  # optional: for debug display
    return results

