import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastembed import TextEmbedding
from google import genai
from google.genai import types

DATA_PATH = Path("data/raw/algiers_medical_places.csv")
INDEX_DIR = Path("data/index")
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

import time

LLM_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Fast, small local embedding model (67MB)
TOP_K = 10

_embed_model: TextEmbedding | None = None


def get_embed_model() -> TextEmbedding:
    """Return a cached local open-source embedding model."""
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model


def get_genai_client() -> genai.Client:
    """Load API key from environment/.env and return a Gemini client."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to .env or export it in your shell."
        )
    return genai.Client(api_key=api_key)


def row_to_chunk(row: pd.Series) -> dict:
    """Convert one CSV row into a structured chunk with OpenStreetMap source metadata."""
    osm_type = str(row["osm_type"]).strip()
    osm_id = str(row["osm_id"]).strip()
    url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}"

    name = None if pd.isna(row["name"]) else str(row["name"]).strip()
    facility_type = str(row["facility_type"]).strip() if pd.notna(row["facility_type"]) else "medical facility"
    title = name or f"Unnamed {facility_type} ({osm_type}/{osm_id})"

    fields = [
        f"Name: {name}" if name else "Name: (unnamed)",
        f"Facility type: {facility_type}",
    ]
    for col, label in [
        ("specialty", "Specialty"),
        ("commune", "Commune"),
        ("address", "Address"),
        ("phone", "Phone"),
        ("website", "Website"),
    ]:
        if pd.notna(row[col]) and str(row[col]).strip():
            fields.append(f"{label}: {str(row[col]).strip()}")

    fields.append(f"Coordinates: {row['latitude']}, {row['longitude']}")
    fields.append(f"Source: {row['source']} ({url})")

    return {
        "id": f"{osm_type}/{osm_id}",
        "title": title,
        "url": url,
        "text": " | ".join(fields),
    }


def build_index() -> None:
    """Read scraped CSV once, embed all records using local open-source model, and save index."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Scraped data not found at {DATA_PATH}. Run scraper.py first.")

    df = pd.read_csv(DATA_PATH)
    chunks = [row_to_chunk(row) for _, row in df.iterrows()]
    texts = [c["text"] for c in chunks]

    print(f"Indexing {len(chunks)} records from {DATA_PATH} using local model '{EMBED_MODEL_NAME}'...")
    embed_model = get_embed_model()
    vectors = list(embed_model.embed(texts))

    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Index successfully built and saved to {INDEX_DIR}/ ({matrix.shape[0]} vectors, dim={matrix.shape[1]}).")


def load_index() -> tuple[np.ndarray, list[dict]]:
    """Load vector store from disk, building it once if it does not exist."""
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        build_index()
    matrix = np.load(EMBEDDINGS_PATH)
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return matrix, chunks


import re

def retrieve(
    query: str,
    matrix: np.ndarray,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[tuple[dict, float]]:
    """Return top-k chunks using hybrid dense semantic retrieval + lexical keyword matching."""
    embed_model = get_embed_model()
    q_vec = list(embed_model.embed([query]))[0].astype(np.float32)
    norm = np.linalg.norm(q_vec)
    if norm > 0:
        q_vec = q_vec / norm

    # Dense semantic cosine similarity
    vector_scores = matrix @ q_vec

    # Lexical keyword boost for named entities, doctors, clinics, and locations
    query_tokens = [w.lower() for w in re.findall(r"[\w]+", query) if len(w) >= 3]
    lexical_scores = np.zeros(len(chunks), dtype=np.float32)
    for i, c in enumerate(chunks):
        title = c["title"].lower()
        text = c["text"].lower()
        for t in query_tokens:
            if t in title:
                lexical_scores[i] += 0.25
            elif t in text:
                lexical_scores[i] += 0.10

    total_scores = vector_scores + lexical_scores
    top_indices = np.argsort(total_scores)[::-1][:top_k]
    return [(chunks[i], float(total_scores[i])) for i in top_indices]


SYSTEM_INSTRUCTION = """You are a helpful Q&A assistant for medical places (doctors and clinics) in Algiers, Algeria, based ONLY on scraped OpenStreetMap records.

Rules you must strictly follow:
1. Answer ONLY using the retrieved records provided in the prompt. Do NOT use outside knowledge or guess missing details.
2. If the retrieved records do not contain enough information to answer the question, refuse gracefully: clearly say that the scraped dataset does not contain that information.
3. Always cite the sources you used by naming the facility/record and including its OpenStreetMap URL from the context.
4. Keep answers clear and concise. Respond in the same language as the user's question when appropriate."""


def generate_with_retry(
    client: genai.Client,
    contents: list[types.Content],
    config: types.GenerateContentConfig,
) -> str:
    """Generate content with automatic retry and model fallback on temporary 503/429 spikes."""
    last_err: Exception | None = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                return (response.text or "").strip()
            except Exception as err:
                last_err = err
                err_str = str(err)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                break  # try next fallback model
    raise RuntimeError(
        f"Gemini API is temporarily overloaded ({last_err}). Please try again in a few seconds."
    )


def answer_question(
    client: genai.Client,
    question: str,
    matrix: np.ndarray,
    chunks: list[dict],
    history: list[types.Content] | None = None,
) -> tuple[str, list[tuple[dict, float]]]:
    """Retrieve relevant chunks and generate a grounded answer using Gemini LLM."""
    results = retrieve(question, matrix, chunks)
    context_lines = [
        f"[{idx}] (similarity={score:.3f}) {chunk['text']}"
        for idx, (chunk, score) in enumerate(results, start=1)
    ]
    user_prompt = (
        "Retrieved medical place records:\n"
        + "\n".join(context_lines)
        + f"\n\nUser question: {question}"
    )

    contents: list[types.Content] = list(history) if history else []
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.1,
    )

    answer = generate_with_retry(client, contents, config)

    if history is not None:
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))

    return answer, results


def print_result(answer: str, results: list[tuple[dict, float]], show_scores: bool = True) -> None:
    """Display the generated answer and retrieved source citations."""
    print(f"\nAnswer:\n{answer}\n")
    if show_scores:
        print("Top retrieved sources:")
        for chunk, score in results[:5]:
            print(f"  - [{score:.3f}] {chunk['title']} -> {chunk['url']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple CLI RAG Chatbot over Algiers Medical Places")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("index", help="Build or rebuild the vector index from the CSV")
    ask_p = sub.add_parser("ask", help="Ask a single question")
    ask_p.add_argument("question", help="Question to ask")
    sub.add_parser("chat", help="Start interactive chat (default)")

    args = parser.parse_args()

    if args.command == "index":
        build_index()
        return

    client = get_genai_client()
    matrix, chunks = load_index()

    if args.command == "ask":
        try:
            answer, results = answer_question(client, args.question, matrix, chunks)
            print_result(answer, results)
        except Exception as err:
            print(f"\n[Error] {err}\n")
        return

    print(f"Loaded {len(chunks)} medical place records. Type 'exit' or 'quit' to stop.\n")
    history: list[types.Content] = []
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        try:
            answer, results = answer_question(client, question, matrix, chunks, history=history)
            print_result(answer, results)
        except Exception as err:
            print(f"\n[Notice] {err}\nPlease ask your question again.\n")


if __name__ == "__main__":
    main()
