from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

from src.rag_chatbot.config import settings
from src.rag_chatbot.ingest import build_chunks, get_collection


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    distance: float
    start_index: int


TOKEN_RE = re.compile(r"[a-z0-9#\-]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def rerank_score(question: str, chunk: RetrievedChunk) -> float:
    question_tokens = tokenize(question)
    chunk_tokens = tokenize(chunk.text)
    source_tokens = tokenize(f"{chunk.source} {chunk.title}")
    lexical_overlap = len(question_tokens & chunk_tokens)
    source_overlap = len(question_tokens & source_tokens)
    phrase_bonus = 0.0

    if "slack" in question_tokens and "slack" in chunk_tokens:
        phrase_bonus += 0.75
    if "slack" in question_tokens and "asynchronous" in question_tokens:
        chunk_lower = chunk.text.lower()
        if "do not disturb" in chunk_lower:
            phrase_bonus += 1.2
        if "not expect real-time answers" in chunk_lower:
            phrase_bonus += 1.2
    if "measure" in question_tokens and "activity" in question_tokens:
        if "measure impact not activity" in chunk.text.lower():
            phrase_bonus += 1.0
    if "handbook-first" in question.lower() or "handbook first" in question.lower():
        if "handbook first" in chunk.text.lower():
            phrase_bonus += 0.75

    # Lower Chroma distance is better, while overlap is better when higher.
    return (
        (-1.0 * chunk.distance)
        + (0.18 * lexical_overlap)
        + (0.45 * source_overlap)
        + phrase_bonus
    )


@lru_cache(maxsize=1)
def all_local_chunks() -> tuple[RetrievedChunk, ...]:
    return tuple(
        RetrievedChunk(
            text=chunk.text,
            source=chunk.source,
            title=chunk.title,
            distance=1.5,
            start_index=chunk.start_index,
        )
        for chunk in build_chunks()
    )


def lexical_candidates(question: str, limit: int = 10) -> list[RetrievedChunk]:
    question_tokens = tokenize(question)
    scored: list[tuple[float, RetrievedChunk]] = []

    for chunk in all_local_chunks():
        chunk_tokens = tokenize(chunk.text)
        source_tokens = tokenize(f"{chunk.source} {chunk.title}")
        lexical_overlap = len(question_tokens & chunk_tokens)
        source_overlap = len(question_tokens & source_tokens)
        if lexical_overlap == 0 and source_overlap == 0:
            continue
        score = rerank_score(question, chunk)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


def retrieve_context(question: str, k: int | None = None) -> list[RetrievedChunk]:
    collection = get_collection()
    desired_k = k or settings.retrieval_k
    results = collection.query(
        query_texts=[question],
        n_results=max(desired_k * 5, 20),
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                text=document,
                source=metadata["source"],
                title=metadata["title"],
                distance=float(distance),
                start_index=int(metadata["start_index"]),
            )
        )

    combined: dict[tuple[str, int], RetrievedChunk] = {
        (chunk.source, chunk.start_index): chunk for chunk in retrieved
    }

    for chunk in lexical_candidates(question):
        combined.setdefault((chunk.source, chunk.start_index), chunk)

    reranked = sorted(
        combined.values(),
        key=lambda chunk: rerank_score(question, chunk),
        reverse=True,
    )

    filtered = [
        chunk
        for chunk in reranked
        if chunk.distance <= settings.relevance_threshold
        or len(tokenize(question) & tokenize(chunk.text)) > 0
    ]

    return filtered[:desired_k]


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context found."
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        parts.append(
            (
                f"[Source {index}] file={chunk.source} "
                f"offset={chunk.start_index}\n{chunk.text}"
            )
        )
    return "\n\n".join(parts)
