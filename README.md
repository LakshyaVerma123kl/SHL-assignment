# 🧠 SHL Intelligent Assessment Recommendation Engine

![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![AI Model](https://img.shields.io/badge/Model-Sentence--Transformers-blue)
![Architecture](https://img.shields.io/badge/Architecture-RAG-purple)
![Compliance](https://img.shields.io/badge/PDF-Strict%20Compliance-green)

A GenAI-powered recommendation system that maps job descriptions to SHL assessments using **Semantic Retrieval-Augmented Generation (RAG)**.

This solution solves the "Keyword Mismatch" problem by using deep learning embeddings (`all-MiniLM-L6-v2`) to understand the _intent_ behind a query (e.g., matching "Team Lead" to "Personality" & "Leadership" assessments) rather than just matching words.

---

## 🚀 Key Features

### 1. 🕷️ Robust Data Ingestion (Selenium)

- **Problem:** The SHL catalog uses infinite scrolling and enterprise-grade firewall protections that block standard scrapers.
- [cite_start]**Solution:** Implemented a **Selenium-based crawler** (`scraper.py`) that mimics human browsing behavior (scrolling, lazy-loading) to successfully build a database of **380+ assessments**[cite: 53, 54].
- **Resilience:** Includes an automated fallback generator to ensure the application never crashes, even if the target site is down.

### 2. 🧠 Semantic RAG Architecture

- **Vectorization:** Assessments are converted into 384-dimensional vectors based on Name, Description, and Category.
- **Retrieval:** Uses **Cosine Similarity** to find the most relevant assessments for any natural language query.

### 3. ⚖️ Balanced Retrieval Algorithm

- [cite_start]**Requirement:** The system must balance recommendations when a query spans multiple domains (e.g., "Java Manager") [cite: 107-109].
- **Implementation:** I developed an **Intent Detection Layer**. If a query implies both _Technical_ and _Behavioral_ needs, the engine switches to an **Interleaving Mode**, ensuring the top results contain a mix of _Knowledge & Skills_ and _Personality_ assessments.

### 4. 📊 Automated Evaluation

- [cite_start]**Metric:** Optimized for **Mean Recall@10**[cite: 106].
- **Tool:** Includes `evaluation.py` to run batch predictions against the training set and calculate accuracy scores automatically.

---

## 🛠️ Project Structure

| File                     | Description                                                                     |
| :----------------------- | :------------------------------------------------------------------------------ |
| `main.py`                | **The Core API.** FastAPI backend with RAG logic and PDF-compliant endpoints.   |
| `scraper.py`             | **The Data Pipeline.** Selenium script to crawl SHL.com and build the database. |
| `evaluation.py`          | **The Judge.** Script to measure Mean Recall@10 against test queries.           |
| `generate_submission.py` | **The Helper.** Generates the `firstname_lastname.csv` for submission.          |
| `generate_pdf.py`        | **The Reporter.** Generates the 2-page technical approach PDF.                  |
| `static/index.html`      | **The Dashboard.** Enterprise UI with Tabular view and CSV export.              |
| `Dockerfile`             | **The Container.** Ensures reproducibility across environments.                 |

---

## ⚡ Setup & Execution

### 1. Prerequisites

- Python 3.9+
- Google Chrome (for the Scraper)

### 2. Installation

```bash
git clone [https://github.com/LakshyaVerma123kl/SHL-assignment](https://github.com/LakshyaVerma123kl/SHL-assignment)
cd shl-assessment-engine
pip install -r requirements.txt
```
