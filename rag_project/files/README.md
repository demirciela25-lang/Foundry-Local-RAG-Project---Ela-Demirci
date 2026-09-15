# Local RAG Assistant (Foundry Local)

An offline, document-grounded Q&A assistant. It stores text chunks and local embeddings
in SQLite, retrieves relevant chunks for each question, and uses Foundry Local to answer.

## Run it

Foundry Local is already installed on this Mac (`foundry --version` reports `0.10.3`).
Open a terminal in this `files` folder, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python ingest.py
python main.py
```

The first ingestion run downloads the embedding model; the first app run downloads the
small chat model. Downloads require internet access; afterwards normal answering is local.

## Add documents

Put UTF-8 `.txt` files in `docs/`, then run `python ingest.py` again. This rebuilds
`local.db`. Three sample documents are included.

Try: `What are the three steps of RAG?`, `Why is SQLite suitable?`, or an unrelated
question such as `What time does the cafeteria close?` (it should decline to guess).

## Troubleshooting

- SDK missing: activate `.venv`, then run `python -m pip install -r requirements.txt`.
- Knowledge base missing: run `python ingest.py` before `python main.py`.
- Model alias unavailable: run `foundry model list`, then set `FOUNDRY_CHAT_MODEL` or
  `FOUNDRY_EMBEDDING_MODEL` before running the app.
