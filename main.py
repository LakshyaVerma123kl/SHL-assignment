import pandas as pd
import re
import io
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import uvicorn

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. THE ENHANCED CATALOGUE ---
# expanded with more realistic SHL test types
assessment_catalog = [
    {
        "id": "shl_001",
        "name": "SHL Verify G+ (General Ability)",
        "description": "The gold standard for cognitive ability. Tests numerical, deductive, and inductive reasoning.",
        "skills": ["problem solving", "critical thinking", "logic", "aptitude", "general intelligence"],
        "min_experience": 0,
        "difficulty": "Medium"
    },
    {
        "id": "shl_002",
        "name": "SHL OPQ32 (Occupational Personality)",
        "description": "Deep behavioral analysis. Measures influence, empathy, structure, and dynamism.",
        "skills": ["leadership", "communication", "culture fit", "sales", "hr", "psychology", "teamwork"],
        "min_experience": 0,
        "difficulty": "N/A"
    },
    {
        "id": "shl_003",
        "name": "SHL Coding: Python/Java Algorithms",
        "description": "Hardcore engineering simulation. Tests syntax, optimization, and bug fixing.",
        "skills": ["python", "java", "backend", "software engineering", "algorithms", "data structures"],
        "min_experience": 1,
        "difficulty": "Hard"
    },
    {
        "id": "shl_004",
        "name": "SHL Front-End (React/Angular)",
        "description": "Browser-based coding tasks involving DOM manipulation, hooks, and CSS styling.",
        "skills": ["react", "javascript", "frontend", "typescript", "css", "html", "web design"],
        "min_experience": 1,
        "difficulty": "Medium"
    },
    {
        "id": "shl_005",
        "name": "SHL Managerial Scenarios (SJT)",
        "description": "Situational judgment for leaders. Focuses on conflict resolution and strategic resource allocation.",
        "skills": ["management", "strategy", "director", "vp", "leadership", "conflict resolution", "planning"],
        "min_experience": 4,
        "difficulty": "Very Hard"
    },
    {
        "id": "shl_006",
        "name": "SHL Numerical Calculation",
        "description": "Speed and accuracy test for data entry, finance, and administrative roles.",
        "skills": ["accounting", "finance", "data entry", "admin", "accuracy", "math", "excel"],
        "min_experience": 0,
        "difficulty": "Easy"
    }
]

# --- 2. THE RECOMMENDATION ENGINE ---
class Recommender:
    def __init__(self):
        self.df = pd.DataFrame(assessment_catalog)
        # Create a rich "soup" of text for matching
        self.df['features'] = (
            self.df['name'] + " " + 
            self.df['description'] + " " + 
            self.df['skills'].apply(lambda x: " ".join(x))
        ).str.lower()
        
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['features'])

    def predict(self, text_input: str, experience: int):
        # Vectorize user input
        user_vec = self.vectorizer.transform([text_input.lower()])
        cosine_sim = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        results = []
        for idx, score in enumerate(cosine_sim):
            row = self.df.iloc[idx]
            final_score = score
            
            # Experience Penalty Logic
            exp_gap = False
            if experience < row['min_experience']:
                final_score *= 0.4  # Heavy penalty for under-qualified
                exp_gap = True

            if final_score > 0.12:  # Threshold to cut noise
                # Explainability: Find which user words matched the catalog
                matched_keywords = [
                    word for word in row['skills'] 
                    if word in text_input.lower()
                ]
                reason = f"Matches skills: {', '.join(matched_keywords[:3])}" if matched_keywords else "Matched based on role description."
                
                results.append({
                    "name": row['name'],
                    "description": row['description'],
                    "match_score": int(round(final_score * 100)),
                    "reason": reason,
                    "warning": f"Requires {row['min_experience']}+ years exp." if exp_gap else None
                })
        
        return sorted(results, key=lambda x: x['match_score'], reverse=True)

# --- 3. THE API SETUP ---
app = FastAPI(title="SHL Architect")
engine = Recommender()

# Enable CORS for production safety
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

class ManualInput(BaseModel):
    role: str
    skills: List[str]
    experience: int

@app.get("/")
def serve_ui():
    return FileResponse('static/index.html')

@app.post("/api/manual")
def recommend_manual(data: ManualInput):
    query = f"{data.role} {' '.join(data.skills)}"
    return engine.predict(query, data.experience)

@app.post("/api/upload")
async def recommend_upload(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        text = " ".join([p.extract_text() for p in reader.pages])
        
        # Smart Heuristic for Experience
        # Looks for 4-digit years (2015-2025) and estimates duration
        years_found = re.findall(r'\b(20[0-2][0-9])\b', text)
        est_exp = 2 # default fallback
        if years_found:
            yrs = sorted([int(y) for y in years_found])
            if len(yrs) > 1:
                est_exp = yrs[-1] - yrs[0]
        
        # Clamp experience to realistic range
        est_exp = max(0, min(est_exp, 25))
        
        recs = engine.predict(text, est_exp)
        
        return {
            "estimated_experience": est_exp,
            "preview": text[:150] + "...",
            "recommendations": recs
        }
        
    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise HTTPException(500, "Could not parse PDF. Ensure it is text-readable.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)