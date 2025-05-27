import csv
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer

def log_results_to_csv(query, entries, filename="search_logs.csv"):
    fieldnames = [
        "timestamp", "query", "title", "query_keywords", "matched_keywords","lexical_match_ratio", "semantic_similarity", 
        "combined_score","passed_semantic", "flan_verified","included","flan_match_quality","response"
    ]
    with open(filename, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        for r in entries:
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "title": r['data']['title'],
                "query_keywords": ", ".join(r.get("query_keywords", [])),
                "matched_keywords": ", ".join(r.get("matched_keywords", [])),
                "lexical_match_ratio": r.get("lexical_match_ratio", None),
                "semantic_similarity": round(r.get("semantic_similarity", None),3),
                "combined_score": round(r.get("combined_score",None),3),
                "passed_semantic": r.get("passed_semantic", None),
                "flan_verified": r.get("flan_verified", None),
                "included": r.get("included", False),
                "flan_match_quality": r.get("match_quality_score",0),
                "response": r.get("flan_response",None)
            })

def extract_keywords(text, n=5):
    vec = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vec.fit_transform([text])
    scores = list(zip(vec.get_feature_names_out(), X.toarray()[0]))
    keywords = [kw for kw, score in sorted(scores, key=lambda x: x[1], reverse=True)]
    return keywords[:n]


def attach_keyword_overlap_metrics(results, rewritten_query):
    query_keywords = extract_keywords(rewritten_query)
    for r in results:
        full_text = " ".join([
            r['data'].get('short_text', ''),
            r['data'].get('description', ''),
            " ".join(r['data'].get('features', []))
        ]).lower()

        matched_keywords = [kw for kw in query_keywords if kw.lower() in full_text]
        r['query_keywords'] = query_keywords
        r['matched_keywords'] = matched_keywords

def enrich_with_scores(results, query_keywords):
    for r in results:
        matched_keywords = [kw for kw in query_keywords if kw in r['data']['combined_text']]
        match_ratio = len(matched_keywords) / len(query_keywords) if query_keywords else 0

        r['query_keywords'] = query_keywords
        r['matched_keywords'] = matched_keywords
        r['lexical_match_ratio'] = match_ratio
        r['semantic_similarity'] = r.get('semantic_similarity', 0)
        r['combined_score'] = 0.6 * r['semantic_similarity'] + 0.4 * match_ratio

    return results