# Algiers Medical Places RAG Chatbot

A Python CLI Retrieval-Augmented Generation (RAG) chatbot that answers questions using **only** scraped OpenStreetMap data of doctors and clinics in Algiers ([algiers_medical_places.csv](data/raw/algiers_medical_places.csv)), complete with source citations, similarity scores, and graceful refusals for out-of-scope questions.

Built for **Challenge 2: Q&A Chatbot over Scraped Data**.

---

## 1. Features & Compliance

- **Grounded Knowledge Base**: Answers questions using only the 821 scraped OpenStreetMap records.
- **One-Time Vector Indexing**: Content is split, structured, embedded, and indexed once into `data/index/`.
- **Fast Local Open-Source Embeddings**: Uses `fastembed` with `BAAI/bge-small-en-v1.5` (67MB, 384 dimensions) to avoid API embedding rate limits and daily quota walls.
- **Gemini LLM**: Powered by `gemini-3.5-flash-lite` with automatic retry and transient spike fallback.
- **Source Citations**: Every answer provides the facility name and direct OpenStreetMap link (e.g. `https://www.openstreetmap.org/node/495024912`).
- **Graceful Refusal**: Accurately declines out-of-scope questions without hallucination.
- **Bonus Capabilities**:
  - Displays cosine similarity scores for top retrieved matches.
  - Multi-turn conversation history support for follow-up questions in interactive chat.

---

## 2. Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure your API key**:
   Copy `.env.example` to `.env` and insert your Google Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
   *(The `.env` file is excluded in `.gitignore` to prevent secret leakage).*

---

## 3. Usage

### 3.1. (Optional) Run the Scraper
The dataset is already generated in `data/raw/algiers_medical_places.csv`. To re-scrape from OpenStreetMap:
```bash
python scraper.py
```

### 3.2. Build the Vector Store (Done Once)
```bash
python rag.py index
```

### 3.3. Interactive CLI Chat
```bash
python rag.py
```
Supports conversation history. Type `exit` or `quit` to end.

### 3.4. Single Question CLI Mode
```bash
python rag.py ask "Where is Clinique Yamina located?"
```

---

## 4. Demo Evaluation (7 Questions)

Run the automated evaluation suite:
```bash
python demo.py
```

### Demo Results Summary

| # | Question | Type | Answer Summary | Sources Cited |
|---|----------|------|----------------|---------------|
| 1 | *Where is Clinique Yamina located?* | Answerable | Rue Bouraada Aissa | [Clinique Yamina (node/495024912)](https://www.openstreetmap.org/node/495024912) |
| 2 | *What is the phone number and address of بوتوشانت طب الداخلي?* | Answerable | Address: Rue des Frères Mokhtari, Phone: 021555223 | [بوتوشانت طب الداخلي (way/1173899099)](https://www.openstreetmap.org/way/1173899099) |
| 3 | *Which medical facility is located in Birtouta?* | Answerable | Etablissement public de santé de Birtouta & polyclinique d'el Kahla | [EPS Birtouta (node/1153109283)](https://www.openstreetmap.org/node/1153109283), [Polyclinique d'el Kahla (way/1173925108)](https://www.openstreetmap.org/way/1173925108) |
| 4 | *Find Dr LARIBI gynécologue and give its location and coordinates.* | Answerable | Address: Bentalha, Coordinates: 36.6487419, 3.0822987 | [Dr LARIBI gynécologue (way/1172857227)](https://www.openstreetmap.org/way/1172857227) |
| 5 | *Where is عيادة القدس located?* | Answerable | شارع إيكزيوم، بلدية حيدرة | [عيادة القدس (node/991924533)](https://www.openstreetmap.org/node/991924533) |
| 6 | *What are the opening hours and consultation fees of Clinique Yamina?* | Out-of-Scope | **Refused gracefully**: The scraped dataset does not contain opening hours or fees. | [Clinique Yamina (node/495024912)](https://www.openstreetmap.org/node/495024912) |
| 7 | *Can you recommend a cardiologist in Oran, Algeria?* | Out-of-Scope | **Refused gracefully**: The scraped dataset only covers Algiers. | Top retrieved Algiers facilities shown with scores |
