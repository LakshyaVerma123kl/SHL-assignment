import pandas as pd
import requests
import json
import logging
from typing import List

# Setup
logging.basicConfig(level=logging.INFO)
API_URL = "http://localhost:8000/recommend"

# Dummy Ground Truth for demonstration (Replace with the actual train.csv provided in the task)
# [cite_start]The PDF says there is a labelled train set with queries and relevant assessments [cite: 58]
TEST_DATA = [
    {
        "query": "Need a Java developer who is good in collaborating with external teams.",
        "expected_keywords": ["Java", "Collaboration", "Teamwork", "Personality"]
    },
    {
        "query": "Hiring a Sales Manager driven by targets.",
        "expected_keywords": ["Sales", "Motivation", "Manager", "Leadership"]
    }
]

def calculate_recall_at_k(predictions: List[str], ground_truth: List[str], k: int = 10) -> float:
    """
    Computes Recall@K: Proportion of relevant items found in the top K recommendations.
    Since we don't have the exact URLs from the hidden dataset, we use keyword matching 
    as a proxy for relevance in this self-evaluation script.
    """
    if not ground_truth: return 0.0
    
    # Consider top K predictions
    top_k_preds = predictions[:k]
    
    # Count how many expected keywords appear in the recommended assessment names/descriptions
    relevant_retrieved = 0
    combined_text = " ".join([p['name'] + " " + p['description'] for p in top_k_preds]).lower()
    
    for truth in ground_truth:
        if truth.lower() in combined_text:
            relevant_retrieved += 1
            
    # Recall = (Relevant Retrieved) / (Total Relevant)
    return relevant_retrieved / len(ground_truth)

def run_evaluation():
    logging.info("🧪 Starting Evaluation (Mean Recall@10)...")
    total_recall = 0
    
    for case in TEST_DATA:
        try:
            # 1. Get Prediction from your API
            response = requests.post(API_URL, json={"query": case["query"]})
            if response.status_code != 200:
                logging.error(f"API Error: {response.text}")
                continue
                
            results = response.json().get("recommended_assessments", [])
            
            # 2. Calculate Score
            recall = calculate_recall_at_k(results, case["expected_keywords"], k=10)
            total_recall += recall
            
            logging.info(f"Query: '{case['query'][:30]}...' | Recall@10: {recall:.2f}")
            
        except Exception as e:
            logging.error(f"Failed to evaluate case: {e}")

    # 3. Final Metric
    mean_recall = total_recall / len(TEST_DATA)
    logging.info(f"✅ Final Score - Mean Recall@10: {mean_recall:.2f}")

if __name__ == "__main__":
    run_evaluation()