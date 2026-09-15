"""Run the Local RAG Assistant from a terminal."""

from rag import RagAssistant


def main() -> None:
    try:
        assistant = RagAssistant()
    except Exception as exc:
        print(f"Setup error: {exc}")
        if "No space left on device" in str(exc) or "errno 28" in str(exc):
            print("Free disk space, then run 'python main.py' again. You do not need to re-run ingestion.")
        return
    print("\nLocal RAG assistant ready. Ask a question, or type 'exit' to quit.\n")
    try:
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            stream, sources = assistant.answer_stream(question)
            print("\nAssistant: ", end="", flush=True)
            print("".join(stream))
            if sources:
                print(f"Sources: {', '.join(sources)}")
            print()
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
