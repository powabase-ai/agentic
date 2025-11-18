"""ETL configuration"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ETLConfig:
    """ETL pipeline configuration"""

    # Extraction
    mistral_api_key: Optional[str] = None
    openai_api_key: str = ""

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"  # recursive, markdown, fixed

    # Embedding (via LiteLLM - supports 100+ providers)
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = (
        "openai"  # openai, cohere, voyage, litellm, sentence-transformers, etc.
    )
