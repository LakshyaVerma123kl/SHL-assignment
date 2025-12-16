import json
import logging
import os
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
# RAG Libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_PATH = "data/shl_assessments.json"
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast & Accurate Semantic Model

# Setup App
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SHL-RAG")
app = FastAPI(title="SHL RAG Engine", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENGINE LOGIC ---
class AssessmentEngine:
    def __init__(self):
        self.assessments = []
        self.embeddings = None
        self.model = None
        
        # Initialize
        self.load_data()
        self.init_model()

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            logger.warning("⚠️ Data file not found! Please run scraper.py. Loading dummy data.")
            self.assessments = [
                {"name": "Python Coding Assessment", "description": "Tests coding skills in Python.", "test_type": ["Knowledge & Skills"], "url": "#", "duration": 45},
                {"name": "OPQ32 Personality", "description": "Workplace behavioral style.", "test_type": ["Personality & Behavior"], "url": "#", "duration": 30}
            ]
        else:
            with open(DATA_PATH, 'r') as f:
                self.assessments = json.load(f)

    def init_model(self):
        logger.info("🧠 Loading Neural Model (Sentence-Transformers)...")
        self.model = SentenceTransformer(MODEL_NAME)
        
        # Create "Context" for each assessment (Name + Description + Type)
        corpus = [
            f"{item['name']} {item['description']} {' '.join(item['test_type'])}" 
            for item in self.assessments
        ]
        self.embeddings = self.model.encode(corpus)
        logger.info("✅ Model Ready.")

    def search(self, query: str, top_k: int = 10):
        # 1. Vectorize Query
        query_vec = self.model.encode([query])
        
        # 2. Semantic Search (Cosine Similarity)
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        
        # 3. Sort Results
        top_indices = np.argsort(scores)[::-1]
        
        # 4. Balanced Retrieval Algorithm (Assignment Requirement)
        # If query implies "Technical" AND "Behavioral", force diversity.
        q_lower = query.lower()
        needs_tech = any(w in q_lower for w in ['java', 'python', 'code', 'technical', 'skill'])
        needs_soft = any(w in q_lower for w in ['lead', 'team', 'communicate', 'behavior', 'manager'])
        
        results = []
        seen = set()
        
        # Buckets for balancing
        tech_picks = []
        soft_picks = []
        
        for idx in top_indices:
            score = float(scores[idx])
            if score < 0.2: continue # Noise filter
            
            item = self.assessments[idx].copy()
            item['score'] = round(score * 100, 1)
            
            # Classify item
            types = " ".join(item['test_type']).lower()
            is_soft = "personality" in types or "behavior" in types
            
            if needs_tech and needs_soft:
                # Force bucket logic
                if is_soft: soft_picks.append(item)
                else: tech_picks.append(item)
            else:
                # Standard Logic
                results.append(item)
                
            if len(results) >= top_k: break

        # Merge buckets if balancing was triggered
        if needs_tech and needs_soft:
            # Interleave results: 1 Tech, 1 Soft, 1 Tech...
            import itertools
            for t, s in itertools.zip_longest(tech_picks, soft_picks):
                if t and t['url'] not in seen: 
                    results.append(t); seen.add(t['url'])
                if s and s['url'] not in seen: 
                    results.append(s); seen.add(s['url'])
        
        return results[:top_k]

# Initialize Engine
engine = AssessmentEngine()

# --- API ENDPOINTS ---
class QueryRequest(BaseModel):
    query: str

@app.post("/recommend")
def recommend(req: QueryRequest):
    recs = engine.search(req.query)
    
    # Format for PDF Requirement
    formatted = []
    for r in recs:
        formatted.append({
            "name": r['name'],
            "url": r['url'],
            "description": r['description'][:200] + "...",
            "test_type": r['test_type'],
            "duration": r.get('duration', 0),
            "match_score": r.get('score', 0)
        })
    return {"recommended_assessments": formatted}

@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}

# Serve Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)