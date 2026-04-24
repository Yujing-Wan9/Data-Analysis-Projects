import argparse
import hashlib
import os
import re
import shutil
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_raw_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return read_txt(path)
    if suffix == ".pdf":
        return read_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def save_processed_text(raw_path: Path, text: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{raw_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = end - overlap

    return chunks


def file_sha256(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def rebuild_processed_dir():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_chroma_collection():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "us_tech_earnings")
    embedding_model = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection


def parse_metadata_from_filename(path: Path) -> dict:
    name = path.stem.lower()

    company = "unknown"
    company_aliases = {
        "microsoft": ["microsoft", "msft"],
        "alphabet": ["alphabet", "goog", "googl", "google"],
        "amazon": ["amazon", "amzn"],
        "meta": ["meta", "facebook"],
        "nvidia": ["nvidia", "nvda"],
        "apple": ["apple", "aapl"],
    }

    for canonical_name, aliases in company_aliases.items():
        if any(alias in name for alias in aliases):
            company = canonical_name
            break

    document_type = "other"
    if "release" in name:
        document_type = "release"
    elif "call" in name or "transcript" in name:
        document_type = "call"
    elif "slide" in name or "presentation" in name:
        document_type = "slides"
    elif "date" in name or "announcement" in name:
        document_type = "date"
    elif "statement" in name:
        document_type = "statement"
    elif "letter" in name:
        document_type = "shareholder_letter"

    return {
        "source_file": path.name,
        "company": company,
        "document_type": document_type,
    }

def rebuild_index():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        [p for p in RAW_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".pdf"}]
    )

    if not raw_files:
        print("No supported files found in data/raw/")
        return

    client, collection = get_chroma_collection()

    # 全量清空 collection
    try:
        client.delete_collection(collection.name)
    except Exception:
        pass
    _, collection = get_chroma_collection()

    total_chunks = 0

    for raw_path in raw_files:
        print(f"Processing: {raw_path.name}")
        raw_text = load_raw_file(raw_path)
        cleaned = clean_text(raw_text)
        processed_path = save_processed_text(raw_path, cleaned)

        chunks = chunk_text(cleaned, chunk_size=800, overlap=120)
        base_metadata = parse_metadata_from_filename(raw_path)
        file_hash = file_sha256(raw_path)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"{raw_path.stem}_chunk_{i}")
            documents.append(chunk)
            md = dict(base_metadata)
            md["chunk_index"] = i
            md["file_hash"] = file_hash
            md["processed_file"] = processed_path.name
            metadatas.append(md)

        if documents:
            collection.add(ids=ids, documents=documents, metadatas=metadatas)
            total_chunks += len(documents)

    print(f"Indexed {len(raw_files)} files and {total_chunks} chunks.")


def main():
    parser = argparse.ArgumentParser(description="Build or update the RAG vector index from data/raw/")
    parser.add_argument("--rebuild", action="store_true", help="Fully rebuild processed texts and vector index")
    args = parser.parse_args()

    load_dotenv()

    if args.rebuild:
        rebuild_processed_dir()

    rebuild_index()


if __name__ == "__main__":
    main()