# 🎯 SHL Assessment Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

An AI-powered system that recommends the best SHL assessments (Verify G+, OPQ32, etc.) based on a candidate's job role, skills, and experience. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to match candidate profiles against a catalog of psychometric tests.

## 🚀 Features

* **🧠 AI Matching Logic:** Uses Scikit-Learn to vectorize text and find semantic similarities between job descriptions and assessment metadata.
* **📄 Resume Parsing:** Built-in PDF parser (using `pypdf`) that extracts text and auto-detects years of experience from resumes.
* **⚡ FastAPI Backend:** High-performance, asynchronous Python API.
* **🎨 Enterprise UI:** Clean, responsive dashboard built with HTML5 and Tailwind CSS (No complex frontend build steps required).
* **🛡️ Experience Guardrails:** Automatically flags or penalizes recommendations if a candidate's experience level is too low for a specific test (e.g., Senior Management SJT).

## 🛠️ Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Processing:** PyPDF (PDF Parsing), Regex
* **Frontend:** HTML5, JavaScript (Vanilla), Tailwind CSS

## 📦 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/LakshyaVerma123kl/SHL-assignment](https://github.com/LakshyaVerma123kl/SHL-assignment)
    cd shl-recommender
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application**
    ```bash
    uvicorn main:app --reload
    ```

5.  **Access the Dashboard**
    Open your browser to `http://localhost:8000`.

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web dashboard (UI). |
| `POST` | `/api/manual` | Accepts JSON (`role`, `skills`, `exp`) and returns ranked recommendations. |
| `POST` | `/api/upload` | Accepts a PDF file, parses text, estimates experience, and returns recommendations. |

## 🧠 How the Recommendation Works

1.  **Vectorization:** The system loads a catalog of SHL assessments (Verify, OPQ, Coding Simulations). It combines the *Name*, *Description*, and *Target Skills* into a single text block.
2.  **TF-IDF Matrix:** It converts this catalog into a mathematical matrix.
3.  **Query Processing:** When a user inputs data (or uploads a resume), the input is vectorized into the same space.
4.  **Cosine Similarity:** The engine calculates the angle between the *User Vector* and every *Assessment Vector*. A smaller angle means a higher match.
5.  **Business Logic:**
    * *Experience Check:* If a user has 2 years exp but the test requires 4, the score is penalized by 60%.
    * *Thresholding:* Matches below 12% relevance are discarded to reduce noise.

## ☁️ Deployment

This project is ready for deployment on **Render** or **Heroku**.
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

---
*Created for the Assessment Tech Innovation Project.*
