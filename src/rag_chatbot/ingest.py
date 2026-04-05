from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.rag_chatbot.config import settings


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    title: str
    start_index: int


def read_markdown_files(directory: Path) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        files.append((path, text))
    return files


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, str]]:
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    chunks: list[tuple[int, str]] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind("\n", start, end)
            if boundary > start + (chunk_size // 2):
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append((start, chunk))
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for path, text in read_markdown_files(settings.handbook_dir):
        rel_path = path.relative_to(settings.handbook_dir).as_posix()
        title = path.stem
        for start_index, chunk_text in split_text(
            text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        ):
            digest = hashlib.md5(f"{rel_path}:{start_index}".encode("utf-8")).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=digest,
                    text=chunk_text,
                    source=rel_path,
                    title=title,
                    start_index=start_index,
                )
            )
    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_documents() -> dict[str, int]:
    chunks = build_chunks()
    collection = get_collection()
    if chunks:
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": chunk.source,
                    "title": chunk.title,
                    "start_index": chunk.start_index,
                }
                for chunk in chunks
            ],
        )
    unique_sources = {chunk.source for chunk in chunks}
    return {
        "documents": len(unique_sources),
        "chunks": len(chunks),
    }


if __name__ == "__main__":
    summary = ingest_documents()
    print(
        f"Ingestion complete. Indexed {summary['documents']} documents and "
        f"{summary['chunks']} chunks into {settings.chroma_dir}."
    )
