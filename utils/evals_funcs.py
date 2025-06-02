"""
evals_funcs.py

This module provides logging utilities for evaluating search behavior in the property matching pipeline.
Logs include final search results, entries dropped at various filtering stages, and FLAN-verified recoveries.

Active Functions:
- log_results_to_csv: Saves top result data for each search query.
- log_faiss_false_negatives: Records listings missed by FAISS but recovered later.
- log_semantic_false_negatives: Records listings filtered by semantic matching but accepted by FLAN.
"""

import csv
import os
from datetime import datetime

def log_results_to_csv(query, entries, filename="search_logs.csv"):
    """
    Logs the final ranked entries returned for a search query.

    Args:
        query (str): The original user search query.
        entries (List[Dict]): Ranked list of matched entries, each with metadata and scores.
        filename (str): Output CSV file (default: "search_logs.csv").

    Each row includes:
        - FAISS rank and score
        - Semantic similarity score
        - FLAN verification result and response
    """

    fieldnames = [
        "timestamp", "query", "title", "faiss_score", "faiss_rank", "semantic_similarity", 
        "passed_semantic", "flan_verified", "flan_response", "flan_match_score"
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
                "semantic_similarity": round(r["semantic_similarity"], 3) if r.get("semantic_similarity") is not None else None,
                "passed_semantic": r.get("passed_semantic", None),
                "flan_response": r.get("flan_response",None),
                "flan_match_score": r.get("flan_match_score"),
                "flan_verified": r.get("flan_verified", None)
            })
            

def log_faiss_false_negatives(recovered_entries: list, filepath: str = "logs/faiss_false_negatives.csv"):

    """
    Logs properties that were rejected by FAISS but recovered by reranking or FLAN verification.

    Args:
        recovered_entries (List[Dict]): List of recovered properties with metadata.
        filepath (str): Output CSV file path.
    """

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
            
    
def log_semantic_false_negatives(query, entries, folder="logs/semantic_false_negatives"):
    """
    Logs properties that were rejected by semantic filtering but verified as true matches by FLAN.

    Args:
        query (str): The original user query.
        entries (List[Dict]): Listings where `passed_semantic` is False but `flan_verified` is True.
        folder (str): Directory to store timestamped CSV logs.
    
    Returns:
        str: Path to the saved log file.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder, f"semantic_misses_{timestamp}.csv")

    fieldnames = [
        "timestamp", "query", "id", "semantic_score", "flan_response", 
        "flan_verified", "title", "short_text", "features", "description"
    ]

    with open(filename, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in entries:
            d = r.get("data", {})
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "id": d.get("id", ""),
                "semantic_score": round(r.get("semantic_similarity", 0), 3),
                "flan_response": r.get("flan_response", ""),
                "flan_verified": r.get("flan_verified", False),
                "title": d.get("title", ""),
                "short_text": d.get("short_text", "").replace("\n", " "),
                "features": " | ".join(d.get("features", [])),
                "description": d.get("description", "").replace("\n", " "),
            })

    return filename

