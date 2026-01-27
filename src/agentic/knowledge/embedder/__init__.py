"""
Embedder module - providers for generating text embeddings.

Embedders transform text into vector embeddings for semantic search.
All embedders use LiteLLM for unified multi-provider support.

Built-in embedders:
- LiteLLMEmbedder: Universal embedder for any LiteLLM-supported provider
- OpenAIEmbedder: OpenAI embedding models (text-embedding-3-small, etc.)
- CohereEmbedder: Cohere embedding models
- VoyageEmbedder: Voyage AI embedding models

Example:
    >>> from agentic.knowledge.embedder import OpenAIEmbedder
    >>>
    >>> embedder = OpenAIEmbedder(model="text-embedding-3-small")
    >>> embedding = embedder.embed("Hello, world!")
    >>> print(len(embedding))  # 1536 dimensions

    >>> # Batch embedding
    >>> embeddings = embedder.embed_batch(["Hello", "World"])
"""

from agentic.knowledge.embedder.base import Embedder
from agentic.knowledge.embedder.litellm import (
    CohereEmbedder,
    LiteLLMEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
)

__all__ = [
    "Embedder",
    "LiteLLMEmbedder",
    "OpenAIEmbedder",
    "CohereEmbedder",
    "VoyageEmbedder",
]
