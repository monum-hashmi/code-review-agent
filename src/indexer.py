import os
import shutil
import tempfile
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from sentence_transformers import SentenceTransformer

from src.config import settings

# Languages we index — everything else is skipped
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "js",
    ".ts": "ts",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}

# Load embedding model once at import time (runs on CPU, no API key)
_embedder = SentenceTransformer(settings.embedding_model if hasattr(settings, 'embedding_model') else "sentence-transformers/all-MiniLM-L6-v2")


def get_chroma_collection():
    """Returns the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    return client.get_or_create_collection(
        name=settings.chroma_collection_name
    )


def index_repo(repo_url: str) -> int:
    """
    Clones repo_url into a temp folder, walks all code files,
    chunks them, embeds with sentence-transformers, stores in ChromaDB.
    Returns total number of chunks stored.
    """
    tmp_dir = tempfile.mkdtemp(prefix=".tmp_repos_")

    try:
        print(f"Cloning {repo_url}...")
        Repo.clone_from(repo_url, tmp_dir)

        collection = get_chroma_collection()
        total_chunks = 0

        for file_path in Path(tmp_dir).rglob("*"):
            if file_path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if any(p in str(file_path) for p in ["venv", "node_modules", ".git", "__pycache__"]):
                continue

            chunks = _chunk_file(file_path, tmp_dir)
            if not chunks:
                continue

            _store_chunks(chunks, collection, repo_url)
            total_chunks += len(chunks)
            print(f"  Indexed {file_path.name} — {len(chunks)} chunks")

        print(f"Done. Total chunks stored: {total_chunks}")
        return total_chunks

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _chunk_file(file_path: Path, repo_root: str) -> list[dict]:
    """
    Reads one file and splits it into overlapping chunks.
    Returns list of {content, metadata} dicts.
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not content.strip():
        return []

    language = SUPPORTED_EXTENSIONS.get(file_path.suffix, "python")
    relative_path = str(file_path).replace(repo_root, "").lstrip("/\\")

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=1000,
        chunk_overlap=200,
    )

    raw_chunks = splitter.split_text(content)

    return [
        {
            "content": chunk,
            "metadata": {
                "file_path": relative_path,
                "language": language,
                "repo_url": "",
                "chunk_index": i,
            }
        }
        for i, chunk in enumerate(raw_chunks)
    ]


def _store_chunks(chunks: list[dict], collection, repo_url: str) -> None:
    """Embeds chunks and stores them in ChromaDB."""
    texts = [c["content"] for c in chunks]
    embeddings = _embedder.encode(texts, show_progress_bar=False).tolist()

    for i, chunk in enumerate(chunks):
        chunk["metadata"]["repo_url"] = repo_url

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"{chunks[i]['metadata']['file_path']}::chunk_{chunks[i]['metadata']['chunk_index']}" for i in range(len(chunks))],
    )


def retrieve_context(query: str, k: int = 5) -> list[str]:
    """
    Given a query (e.g. a function name from the diff),
    returns top-k most relevant code chunks from ChromaDB.
    This is what agents call to get codebase context.
    """
    collection = get_chroma_collection()
    query_embedding = _embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )

    return results["documents"][0] if results["documents"] else []