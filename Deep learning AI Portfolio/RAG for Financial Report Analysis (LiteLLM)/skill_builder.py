# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 17:05:15 2026

@author: User
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv


GLOBAL_QUESTIONS = {
    "overview": "請用 200 字內總結這個知識庫。請明確說明：這是一個以美股大型科技公司財報、法說會逐字稿、新聞稿與投資人關係文件為主的知識庫；涵蓋哪些公司；主要用途是什麼；不要把 AI 說成整個知識庫唯一主題。",
    "core_concepts": "請整理這個知識庫中最重要的 8 到 12 個核心概念，每點用簡短條列並附 1 句說明。",
    "key_trends": "請整理這個知識庫目前最重要的 5 到 8 個趨勢或發展方向。",
    "key_entities": "請整理這個知識庫中的重要實體，並務必先列出所有主要公司（若有出現 Apple、Microsoft、Alphabet、Amazon、Meta、NVIDIA 請明確列出），再列產品/業務、技術主題、文件類型。不要加入與財報主題無關的零碎名詞。",
    "methodology": "請整理從這個知識庫可觀察到的分析方法、閱讀重點或最佳實踐，例如看財報時常關注哪些面向。",
    "limitations": "請說明這個知識庫的知識邊界與限制，例如涵蓋公司、季度、文件類型、可能缺漏的主題。",
    "example_qa": "請生成 4 組代表性問答，問題要像使用者真的會問的，答案要簡潔。",
}


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


def retrieve_context(query: str, top_k: int = 8):
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas


def format_context(docs: List[str], metas: List[Dict]) -> str:
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        parts.append(
            f"[Source {i}] "
            f"company={meta.get('company', 'unknown')}, "
            f"type={meta.get('document_type', 'unknown')}, "
            f"file={meta.get('source_file', 'unknown')}, "
            f"chunk={meta.get('chunk_index', -1)}\n{doc}"
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


def answer_global_question(
    question: str,
    top_k: int,
    model: str | None,
    metadata_hint: str = ""
) -> str:
    docs, metas = retrieve_context(question, top_k=top_k)
    context = format_context(docs, metas)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial RAG summarization assistant. "
                "Answer only based on the provided context. "
                "If the context is insufficient, say so clearly. "
                "Respond in Traditional Chinese."
            ),
        },
        {
            "role": "user",
            "content": f"{metadata_hint}\n\nQuestion:\n{question}\n\nContext:\n{context}",
        },
    ]
    return call_llm(messages, model_override=model)


def summarize_metadata() -> Dict[str, List[str]]:
    collection = get_collection()
    data = collection.get(include=["metadatas"])
    metas = data.get("metadatas", [])

    companies = sorted({m.get("company", "unknown") for m in metas if m.get("company")})
    doc_types = sorted(
        {m.get("document_type", "other") for m in metas if m.get("document_type")}
    )
    source_files = sorted(
        {m.get("source_file", "unknown") for m in metas if m.get("source_file")}
    )

    return {
        "companies": companies,
        "doc_types": doc_types,
        "source_files": source_files,
    }


def build_skill_markdown(model: str | None, top_k: int) -> str:
    metadata_summary = summarize_metadata()
    companies_text = ", ".join(metadata_summary["companies"])
    doc_types_text = ", ".join(metadata_summary["doc_types"])
    metadata_hint = f"""
已知這個知識庫的公司包含：{companies_text}
已知文件類型包含：{doc_types_text}
請在總結時優先以這些已知 metadata 為基礎，不要忽略已出現的公司，也不要把局部 chunks 的主題誤當成整個知識庫唯一主題。
"""

    overview = answer_global_question(GLOBAL_QUESTIONS["overview"], top_k, model, metadata_hint)
    core_concepts = answer_global_question(GLOBAL_QUESTIONS["core_concepts"], top_k, model, metadata_hint)
    key_trends = answer_global_question(GLOBAL_QUESTIONS["key_trends"], top_k, model, metadata_hint)
    key_entities = answer_global_question(GLOBAL_QUESTIONS["key_entities"], top_k, model, metadata_hint)
    methodology = answer_global_question(GLOBAL_QUESTIONS["methodology"], top_k, model, metadata_hint)
    limitations = answer_global_question(GLOBAL_QUESTIONS["limitations"], top_k, model, metadata_hint)
    example_qa = answer_global_question(GLOBAL_QUESTIONS["example_qa"], top_k, model, metadata_hint)

    companies_text = ", ".join(metadata_summary["companies"])
    doc_types_text = ", ".join(metadata_summary["doc_types"])
    total_sources = len(metadata_summary["source_files"])
    today = datetime.now().strftime("%Y-%m-%d")

    source_reference_lines = "\n".join(
        f"- {fname}" for fname in metadata_summary["source_files"]
    )

    markdown = f"""# Skill: U.S. Big Tech Earnings and Investor Communications

## Metadata
- **知識領域**：美股科技公司財報與法說會分析
- **資料來源數量**：{total_sources} 份文件
- **最後更新時間**：{today}
- **適用 Agent 類型**：研究助手 / 投資研究問答機器人 / 財報分析助手

## Overview
{overview}

## Core Concepts
{core_concepts}

## Key Trends
{key_trends}

## Key Entities
{key_entities}

## Methodology & Best Practices
{methodology}

## Knowledge Gaps & Limitations
{limitations}

## Example Q&A
{example_qa}

## Source References
- 公司涵蓋：{companies_text}
- 文件類型：{doc_types_text}
{source_reference_lines}
"""
    return markdown


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate skill.md from the RAG knowledge base")
    parser.add_argument("--output", type=str, default="skill.md", help="Output markdown filename")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks retrieved per global question")
    args = parser.parse_args()

    markdown = build_skill_markdown(model=args.model, top_k=args.top_k)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"Skill file written to {args.output}")


if __name__ == "__main__":
    main()