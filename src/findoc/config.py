"""Settings, overridable through FINDOC_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = Path("data")

    # Ingestion
    ocr_lang: str = "eng"
    ocr_dpi: int = 300
    ocr_min_chars: int = 40  # a page with images and less text than this gets OCRed
    tesseract_cmd: str = ""

    # Chunking
    chunk_chars: int = 1200
    chunk_overlap: int = 150

    # Retrieval
    store: str = "faiss"  # faiss | qdrant
    qdrant_url: str = ""  # empty means embedded local mode under data_dir
    qdrant_collection: str = "findoc"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 6

    # LLM
    llm_backend: str = "transformers"  # transformers | openai
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_adapter: str = ""  # path to a trained LoRA adapter (transformers backend)
    llm_load_in_4bit: bool = False
    llm_base_url: str = "http://localhost:11434/v1"  # Ollama, vLLM, llama.cpp server
    llm_api_key: str = "not-needed"
    max_new_tokens: int = 400

    @classmethod
    def from_env(cls) -> "Settings":
        settings = cls()
        for field in fields(cls):
            raw = os.environ.get(f"FINDOC_{field.name.upper()}")
            if raw is None:
                continue
            current = getattr(settings, field.name)
            if isinstance(current, bool):
                value = raw.strip().lower() in {"1", "true", "yes", "on"}
            else:
                value = type(current)(raw)
            setattr(settings, field.name, value)
        return settings
