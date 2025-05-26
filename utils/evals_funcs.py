import csv
import os
from datetime import datetime

def log_results_to_csv(query, entries, filename="search_logs.csv"):
    fieldnames = [
        "timestamp", "query", "title", "rerank_score", "semantic_similarity",
        "lexical_score", "entailment", "contradiction",
        "passed_semantic", "passed_lexical", "included"
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
                "rerank_score": r.get("rerank_score", None),
                "semantic_similarity": r.get("semantic_similarity", None),
                "lexical_score": r.get("lexical_score", None),
                "entailment": r.get("nli_scores", {}).get("entailment", None),
                "contradiction": r.get("nli_scores", {}).get("contradiction", None),
                "passed_semantic": r.get("passed_semantic", None),
                "passed_lexical": r.get("passed_lexical", None),
                "included": r.get("included", False)
            })

