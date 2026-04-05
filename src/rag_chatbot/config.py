from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    handbook_dir: Path = Path(
        os.getenv("HANDBOOK_DIR", str(ROOT_DIR / "data" / "handbook"))
    )
    chroma_dir: Path = ROOT_DIR / "chroma_db"
    evaluation_dir: Path = ROOT_DIR / "evaluation_results"
    golden_dataset_path: Path = ROOT_DIR / "data" / "golden_dataset.json"
    collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "gitlab_handbook")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "1.2"))
    litellm_model: str = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-flash")
    litellm_api_key: str = os.getenv("LITELLM_API_KEY", "") or os.getenv(
        "GOOGLE_API_KEY", ""
    ) or os.getenv(
        "GEMINI_API_KEY", ""
    )
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        (
            "You are a helpful RAG assistant. Use only the provided context. "
            "If the answer is not supported by the context, say that you cannot "
            "find the answer in the knowledge base."
        ),
    )


settings = Settings()
