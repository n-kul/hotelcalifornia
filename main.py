import sqlite3
import base64
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx

# --- Fill your keys here ---
AIPROXY_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDEwMDhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9._MTfkLWLurT3DLStyzoHKg8c2ugWNGUpfCDLmnti6gg"
AIPROXY_EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

AIPIPE_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjIwMDEwMDhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.WIFq02daMudpp5TUxX6FFSLY9jfh0gspWZ-__8J6adM"
AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"

DB_PATH = "knowledge_base.db"

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionPayload(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[Link]

# Helper: Decode embedding from BLOB (assumes JSON stored as bytes)
def blob_to_embedding(blob) -> np.ndarray:
    try:
        arr = json.loads(blob.decode("utf-8"))
        return np.array(arr, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Invalid embedding format: {e}")

# Cosine similarity calculation
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get top k similar chunks from DB by embedding similarity
def get_top_k_chunks(question_embedding: np.ndarray, k=3):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    all_chunks = []

    cursor.execute("SELECT content, url, embedding FROM discourse_chunks WHERE embedding IS NOT NULL")
    for content, url, emb_blob in cursor.fetchall():
        emb = blob_to_embedding(emb_blob)
        score = cosine_similarity(question_embedding, emb)
        all_chunks.append({"content": content, "url": url, "score": score})

    cursor.execute("SELECT content, original_url, embedding FROM markdown_chunks WHERE embedding IS NOT NULL")
    for content, url, emb_blob in cursor.fetchall():
        emb = blob_to_embedding(emb_blob)
        score = cosine_similarity(question_embedding, emb)
        all_chunks.append({"content": content, "url": url, "score": score})

    conn.close()

    all_chunks.sort(key=lambda x: x["score"], reverse=True)
    return all_chunks[:k]

# Get embedding for question using AIProxy
async def get_question_embedding(question: str) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {AIPROXY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": question,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(AIPROXY_EMBEDDING_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return np.array(data["data"][0]["embedding"], dtype=np.float32)

# Query AIPipe chat completion API
async def query_aipipe_chat(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(AIPIPE_CHAT_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: QuestionPayload):
    try:
        if payload.image:
            base64.b64decode(payload.image)

        # 1. Get embedding vector of question
        question_embedding = await get_question_embedding(payload.question)

        # 2. Retrieve top-k relevant chunks from DB
        top_chunks = get_top_k_chunks(question_embedding, k=3)
        if not top_chunks:
            return {"answer": "No relevant context found.", "links": []}

        # 3. Construct prompt with context and question
        context = "\n\n".join([chunk["content"] for chunk in top_chunks])
        prompt = f"Context:\n{context}\n\nQuestion:\n{payload.question}\nAnswer:"

        # 4. Get answer from AIPipe chat
        answer = await query_aipipe_chat(prompt)

        # 5. Format source links for response
        links = [
            {
                "url": chunk["url"] or "N/A",
                "text": (chunk["content"][:150] + "...") if len(chunk["content"]) > 150 else chunk["content"],
            }
            for chunk in top_chunks
        ]

        return {"answer": answer, "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
