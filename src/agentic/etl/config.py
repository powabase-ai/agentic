"""ETL configuration"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ETLConfig:
    """ETL pipeline configuration"""

    # Extraction - PDFs
    mistral_api_key: Optional[str] = None
    openai_api_key: str = ""

    # Extraction - Images (vision models)
    vision_model: str = (
        "gpt-4o"  # gpt-4o, gpt-4o-mini, claude-3-sonnet, gemini-pro-vision, etc.
    )
    vision_max_tokens: int = 1000
    vision_temperature: float = 0.7

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"  # recursive, markdown, fixed

    # Embedding (via LiteLLM - supports 100+ providers)
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = (
        "openai"  # openai, cohere, voyage, litellm, sentence-transformers, etc.
    )
