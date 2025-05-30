import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from difflib import SequenceMatcher


def log_results_to_csv(query, entries, filename="search_logs.csv"):
    fieldnames = [
        "timestamp", "query", "title", "faiss_score", "faiss_rank","query_keywords", "matched_keywords","lexical_match_ratio", "semantic_similarity", 
        "combined_score","passed_semantic", "flan_verified", "flan_response", "flan_match_score"
    ]
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for r in entries:
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "faiss_rank": r['data']['faiss_rank'],
                "faiss_score": r['data']['faiss_score'],
                "title": r['data']['title'],
                "query_keywords": ", ".join(r.get("query_keywords", [])),
                "matched_keywords": ", ".join(r.get("matched_keywords", [])),
                "lexical_match_ratio": r.get("lexical_match_ratio", None),
                "semantic_similarity": round(r.get("semantic_similarity", None),3),
                "combined_score": round(r.get("combined_score",None),3),
                "passed_semantic": r.get("passed_semantic", None),
                "flan_response": r.get("flan_response",None),
                "flan_match_score": r.get("flan_match_score"),
                "flan_verified": r.get("flan_verified", None)
            })
            

def log_faiss_false_negatives(recovered_entries: list, filepath: str = "logs/faiss_false_negatives.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'title', 'faiss_score', 'flan_response', 'short_text', 'description', 'features'
        ])
        writer.writeheader()

        for entry in recovered_entries:
            data = entry['data']
            writer.writerow({
                'id': data.get('id'),
                'title': data.get('title', 'N/A'),
                'faiss_score': entry['score'],
                'flan_response': entry.get('flan_response', 'N/A'),
                'short_text': data.get('short_text', '').replace('\n', ' '),
                'description': data.get('description', '').replace('\n', ' '),
                'features': " | ".join(data.get('features', []))
            })
            

def extract_keywords(text, n=5):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vec.fit_transform([text])
    scores = list(zip(vec.get_feature_names_out(), X.toarray()[0]))
    keywords = [kw for kw, score in sorted(scores, key=lambda x: x[1], reverse=True)]
    return keywords[:n]


def fuzzy_contains(phrase, text, threshold=0.77, max_window_size=5):
    phrase = phrase.lower()
    phrase_len = len(phrase.split())
    
        # Make threshold stricter for short phrases
    if phrase_len <= 2 and threshold < 0.85:
        threshold = 0.85

    
    words = text.lower().split()

    for i in range(len(words)):
        for j in range(i + 1, min(len(words) + 1, i + max_window_size + 1)):
            if abs(j - i - phrase_len) > 1:
                continue  # optional: skip unrelated lengths

            window = " ".join(words[i:j])
            score = SequenceMatcher(None, phrase, window).ratio()
            if score >= threshold:
                return True
    return False


def attach_keyword_overlap_metrics(results, semantic_focus):
    query_keywords = extract_keywords(semantic_focus)

    for r in results:
        full_text = " ".join([
            r['data'].get('short_text', ''),
            r['data'].get('description', ''),
            " ".join(r['data'].get('features', []))
        ]).lower()

        matched_keywords = []
        for kw in query_keywords:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            if re.search(pattern, full_text) or fuzzy_contains(kw.lower(), full_text):
                matched_keywords.append(kw)

        r['query_keywords'] = query_keywords
        r['matched_keywords'] = matched_keywords


    
    
def enrich_with_scores(results, query_keywords=None):
    for r in results:
        # Use already extracted keywords
        q_keywords = r.get("query_keywords", query_keywords or [])
        m_keywords = r.get("matched_keywords", [])

        match_ratio = len(m_keywords) / len(q_keywords) if q_keywords else 0

        r['lexical_match_ratio'] = match_ratio
        r['semantic_similarity'] = r.get('semantic_similarity', 0)
        r['combined_score'] = 0.6 * r['semantic_similarity'] + 0.4 * match_ratio

    return results

