"""Retrieve document chunks from SQLite and generate grounded local answers."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Iterator

from foundry_setup import load_chat_client, load_embedding_client, unload_models

DB_PATH = Path(__file__).resolve().parent / "local.db"
TOP_K = 3
MIN_RELEVANCE = 0.20
FALLBACK_ANSWER = "I don't have that information in my documents."
SYSTEM_PROMPT = (
    "Answer using only the supplied document context. Do not use outside knowledge or invent facts. "
    f"If the context does not answer the question, reply exactly: '{FALLBACK_ANSWER}' Keep it concise."
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


class RagAssistant:
    def __init__(self) -> None:
        if not DB_PATH.exists():
            raise FileNotFoundError("Knowledge base not found. Run 'python ingest.py' first.")
        print("Loading local models; the first run may download them...")
        self.embedding_client = load_embedding_client()
        self.chat_client = load_chat_client()

    def close(self) -> None:
        unload_models()

    def get_top_chunks(self, query: str, k: int = TOP_K) -> list[tuple[float, str, int, str]]:
        with sqlite3.connect(DB_PATH) as connection:
            rows = connection.execute("SELECT source, chunk_number, content, embedding FROM chunks").fetchall()
        if not rows:
            return []
        query_embedding = self.embedding_client.generate_embedding(query).data[0].embedding
        scores = [
            (cosine_similarity(query_embedding, json.loads(embedding)), source, number, content)
            for source, number, content, embedding in rows
        ]
        return sorted(scores, reverse=True, key=lambda item: item[0])[:k]

    def answer_stream(self, question: str) -> tuple[Iterator[str], list[str]]:
        matches = [match for match in self.get_top_chunks(question) if match[0] >= MIN_RELEVANCE]
        if not matches:
            return iter((FALLBACK_ANSWER,)), []
        context = "\n\n".join(
            f"[Source: {source}, chunk {number}]\n{content}"
            for _, source, number, content in matches
        )
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}"},
            {"role": "user", "content": question},
        ]
        def tokens() -> Iterator[str]:
            for chunk in self.chat_client.complete_streaming_chat(messages):
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        sources = sorted({f"{source} (chunk {number})" for _, source, number, _ in matches})
        return tokens(), sources
