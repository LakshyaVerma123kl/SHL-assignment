import json
import logging
import os
import numpy as np
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
DATA_PATH = "data/shl_assessments.json"
MODEL_NAME = 'all-MiniLM-L6-v2'

# Setup Logging & App
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SHL-API")
app = FastAPI(title="SHL Assessment API", version="3.0 (PDF Compliant)")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STRICT MODELS (Matches Appendix 2 of PDF) ---
class HealthResponse(BaseModel):
    status: str

class QueryRequest(BaseModel):
    query: str

class AssessmentItem(BaseModel):
    # These exact fields are required by the assignment PDF
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: int
    remote_support: str
    test_type: List[str]

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentItem]

# --- INTELLIGENT ENGINE ---
class AssessmentEngine:
    def __init__(self):
        self.assessments = []
        self.embeddings = None
        self.model = None
        self.load_data()
        self.init_model()

    def load_data(self):
        if not os.path.exists(DATA_PATH):
            logger.error("❌ Data file missing! Run 'python scraper.py' first.")
            self.assessments = []
        else:
            with open(DATA_PATH, 'r') as f:
                self.assessments = json.load(f)

    def init_model(self):
        logger.info("🧠 Loading AI Model...")
        self.model = SentenceTransformer(MODEL_NAME)
        corpus = [
            f"{item['name']} {item['description']} {' '.join(item['test_type'])}" 
            for item in self.assessments
        ]
        if corpus:
            self.embeddings = self.model.encode(corpus)
            logger.info("✅ Embeddings ready.")

    def search(self, query: str, top_k: int = 10):
        if not self.assessments or self.embeddings is None:
            return []

        # 1. Semantic Search
        query_vec = self.model.encode([query])
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1]

        # 2. Balanced Retrieval (Technical vs. Behavioral)
        q_lower = query.lower()
        needs_tech = any(w in q_lower for w in ['java', 'python', 'code', 'technical', 'skill', 'sql'])
        needs_soft = any(w in q_lower for w in ['lead', 'team', 'communicate', 'behavior', 'manager'])
        
        results = []
        seen_urls = set()
        
        # Buckets for balancing
        tech_picks = []
        soft_picks = []

        for idx in top_indices:
            if scores[idx] < 0.2: continue 
            item = self.assessments[idx]
            
            # Simple classifier for balancing logic
            types = " ".join(item['test_type']).lower()
            is_soft = "personality" in types or "behavior" in types or "biodata" in types
            
            if needs_tech and needs_soft:
                if is_soft: soft_picks.append(item)
                else: tech_picks.append(item)
            else:
                if item['url'] not in seen_urls:
                    results.append(item)
                    seen_urls.add(item['url'])
            
            if len(results) >= top_k: break

        # Interleave if query needs balance
        if needs_tech and needs_soft:
            import itertools
            for t, s in itertools.zip_longest(tech_picks, soft_picks):
                if t and t['url'] not in seen_urls:
                    results.append(t); seen_urls.add(t['url'])
                if s and s['url'] not in seen_urls:
                    results.append(s); seen_urls.add(s['url'])
        
        return results[:top_k]

engine = AssessmentEngine()

# --- ENDPOINTS (Strictly matching PDF Appendix 2) ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(request: QueryRequest):
   
    recs = engine.search(request.query)
    
    formatted = []
    for r in recs:
        formatted.append(AssessmentItem(
            url=r['url'],
            name=r['name'],
            # Ensure these fields exist even if scraper missed them
            adaptive_support=r.get('adaptive_support', 'No'),
            description=r.get('description', 'No description available')[:400], # Limit length
            duration=r.get('duration', 0),
            remote_support=r.get('remote_support', 'Yes'),
            test_type=r.get('test_type', ["General"])
        ))
    
    return {"recommended_assessments": formatted}

# Serve Frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)