import argparse
import os
from typing import Dict, List

from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the RAG knowledge base")
    parser.add_argument("--query", type=str, help="Single query mode")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name, e.g. gemini-2.5-flash or gpt-oss:20b",
    )
    return parser


def get_collection():
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "us_tech_earnings")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )
    return collection


def retrieve_context(query: str, top_k: int):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return docs, metas


def format_context(docs: List[str], metas: List[Dict]) -> str:
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        source = meta.get("source_file", "unknown")
        company = meta.get("company", "unknown")
        doc_type = meta.get("document_type", "unknown")
        chunk_index = meta.get("chunk_index", -1)

        parts.append(
            f"[Source {i}] company={company}, type={doc_type}, "
            f"file={source}, chunk={chunk_index}\n{doc}"
        )
    return "\n\n".join(parts)


def call_llm(messages: List[Dict[str, str]], model_override: str | None = None) -> str:
    from openai import OpenAI

    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")
    model = model_override or os.getenv("LITELLM_MODEL", "gemini-2.5-flash")

    if not api_key or not base_url:
        raise RuntimeError("Missing LITELLM_API_KEY or LITELLM_BASE_URL in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


def print_sources(metas: List[Dict]) -> None:
    print("\n=== Sources ===")
    for i, meta in enumerate(metas, start=1):
        print(
            f"{i}. file={meta.get('source_file')} | "
            f"company={meta.get('company')} | "
            f"type={meta.get('document_type')} | "
            f"chunk={meta.get('chunk_index')}"
        )


def single_query_mode(query: str, top_k: int, model: str | None) -> None:
    docs, metas = retrieve_context(query, top_k)
    context = format_context(docs, metas)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial document RAG assistant. "
                "Answer only based on the provided context. "
                "If the context is insufficient, say so clearly. "
                "Respond in Traditional Chinese. "
                "At the end, briefly mention the most relevant sources."
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{query}\n\nContext:\n{context}",
        },
    ]

    try:
        answer = call_llm(messages, model_override=model)
        print("\n=== Answer ===")
        print(answer)
    except Exception as e:
        print("\n=== LLM unavailable, showing retrieved context only ===")
        print(f"Reason: {e}")
        print("\n=== Retrieved Context ===")
        print(context)

    print_sources(metas)


def interactive_mode(top_k: int, model: str | None) -> None:
    print("Entering interactive mode. Type 'exit' to quit.")

    history: List[Dict[str, str]] = []

    while True:
        query = input("\nQuestion> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        docs, metas = retrieve_context(query, top_k)
        context = format_context(docs, metas)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial document RAG assistant. "
                    "Answer only based on the provided context. "
                    "If the context is insufficient, say so clearly. "
                    "Respond in Traditional Chinese."
                ),
            }
        ]

        if history:
            messages.extend(history[-6:])  # 保留最近 3 輪

        messages.append(
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nContext:\n{context}",
            }
        )

        try:
            answer = call_llm(messages, model_override=model)
            print("\n=== Answer ===")
            print(answer)

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})
        except Exception as e:
            print("\n=== LLM unavailable, showing retrieved context only ===")
            print(f"Reason: {e}")
            print("\n=== Retrieved Context ===")
            print(context)

        print_sources(metas)


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    if args.query:
        single_query_mode(args.query, args.top_k, args.model)
    else:
        interactive_mode(args.top_k, args.model)


if __name__ == "__main__":
    main()