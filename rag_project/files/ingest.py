"""Build the SQLite knowledge base from text files in ``docs/``."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from foundry_setup import load_embedding_client, unload_models

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "local.db"
DOCS_DIR = PROJECT_DIR / "docs"
DEFAULT_MAX_CHARS = 700
BATCH_SIZE = 32


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Split text into readable, bounded chunks without discarding long text."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
        for piece in pieces:
            proposed = f"{current}\n\n{piece}" if current else piece
            if current and len(proposed) > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = proposed
    if current:
        chunks.append(current)
    return chunks


def document_chunks(docs_dir: Path, max_chars: int) -> list[tuple[str, int, str]]:
    """Return (source filename, chunk number, content) for each text document."""
    records: list[tuple[str, int, str]] = []
    for path in sorted(docs_dir.glob("*.txt")):
        for index, content in enumerate(chunk_text(path.read_text(encoding="utf-8"), max_chars), 1):
            records.append((path.name, index, content))
    return records


def create_database(records: list[tuple[str, int, str]], embeddings: list[list[float]]) -> None:
    if len(records) != len(embeddings):
        raise ValueError("The embedding service returned an unexpected number of vectors.")
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("DROP TABLE IF EXISTS chunks")
        connection.execute(
            "CREATE TABLE chunks (id INTEGER PRIMARY KEY, source TEXT NOT NULL, "
            "chunk_number INTEGER NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO chunks (source, chunk_number, content, embedding) VALUES (?, ?, ?, ?)",
            [(source, number, content, json.dumps(vector))
             for (source, number, content), vector in zip(records, embeddings)],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index .txt documents into the local RAG database.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()
    if not DOCS_DIR.exists():
        print(f"No docs directory at {DOCS_DIR}. Create it and add .txt files.")
        return
    records = document_chunks(DOCS_DIR, args.chunk_size)
    if not records:
        print(f"No .txt files found in {DOCS_DIR}. Add documents and run this again.")
        return
    print(f"Found {len(records)} chunk(s). Loading the embedding model...")
    try:
        client = load_embedding_client()
        embeddings: list[list[float]] = []
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            response = client.generate_embeddings([content for _, _, content in batch])
            embeddings.extend(item.embedding for item in response.data)
            print(f"Embedded {min(start + len(batch), len(records))}/{len(records)} chunks.")
        create_database(records, embeddings)
    finally:
        unload_models()
    print(f"Done. Indexed {len(records)} chunks into {DB_PATH.name}.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Setup error: {exc}")
