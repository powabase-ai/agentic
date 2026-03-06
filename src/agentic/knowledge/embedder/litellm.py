"""
LiteLLM Embedder - unified interface for multiple embedding providers.

Supports all providers via LiteLLM:
- OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- Cohere: embed-english-v3.0, embed-multilingual-v3.0
- Vertex AI: textembedding-gecko, text-embedding-004
- Bedrock: amazon.titan-embed-text-v1
- Mistral: mistral-embed
- Voyage: voyage-01, voyage-lite-01
- HuggingFace: any model via huggingface/ prefix

Adapted from proven implementation in agentic/etl/transformers/embedders/litellm.py.
"""

import asyncio
import logging
from typing import Any

from agentic.knowledge.embedder.base import Embedder
from agentic.knowledge.model_config import CHUNK_EMBED_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class LiteLLMEmbedder(Embedder):
    """
    Universal embedder using LiteLLM for multi-provider support.

    This embedder supports all major embedding providers through a single
    interface, with automatic batching, retries, and error handling.

    Args:
        model: Model identifier (e.g., 'text-embedding-3-small', 'cohere/embed-english-v3.0')
        api_key: Optional API key (defaults to environment variable)
        api_base: Optional custom API endpoint
        dimensions: Optional embedding dimensions (supported by some models)
        timeout: Request timeout in seconds (default: 600)
        **provider_params: Provider-specific parameters (e.g., input_type for Cohere)

    Example:
        >>> embedder = LiteLLMEmbedder(model="text-embedding-3-small")
        >>> embedding = embedder.embed("Hello world")
        >>> print(len(embedding))  # 1536 for text-embedding-3-small

        >>> # Batch embedding
        >>> embeddings = embedder.embed_batch(["Hello", "World"])
        >>> print(len(embeddings))  # 2
    """

    name = "litellm"

    def __init__(
        self,
        model: str = CHUNK_EMBED_EMBEDDING_MODEL,
        api_key: str | None = None,
        api_base: str | None = None,
        dimensions: int | None = None,
        timeout: int = 600,
        **provider_params: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._dimensions = dimensions
        self.timeout = timeout
        self.provider_params = provider_params

        logger.info(f"Initialized LiteLLM embedder with model: {model}")

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for the model."""
        if self._dimensions:
            return self._dimensions

        # Common model dimensions
        known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
        }

        for pattern, dims in known_dimensions.items():
            if pattern in self.model:
                return dims

        # Default fallback
        return 1536

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            from litellm import embedding as litellm_embedding
        except ImportError:
            raise ImportError(
                "litellm is required for LiteLLMEmbedder. "
                "Install with: pip install litellm"
            ) from None

        if not texts:
            return []

        # Build kwargs for litellm
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "timeout": self.timeout,
        }

        # Add optional parameters
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions

        # Add provider-specific params
        kwargs.update(self.provider_params)

        try:
            response = litellm_embedding(**kwargs)
            embeddings = [item["embedding"] for item in response["data"]]

            logger.debug(
                f"Generated {len(embeddings)} embeddings using {self.model} "
                f"(tokens: {response.get('usage', {}).get('total_tokens', 'unknown')})"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed with {self.model}: {e}")
            raise

    async def aembed(self, text: str) -> list[float]:
        """
        Async version of embed (implements ABC).

        Uses asyncio to run the sync embedding in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed, text)

    # Alias for backwards compatibility
    async_embed = aembed

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Async version of embed_batch (implements ABC).

        Uses asyncio to run the sync embedding in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)

    # Alias for backwards compatibility
    async_embed_batch = aembed_batch


class OpenAIEmbedder(LiteLLMEmbedder):
    """
    OpenAI embedder (convenience alias).

    Example:
        >>> embedder = OpenAIEmbedder()  # Uses text-embedding-3-small
        >>> embedder = OpenAIEmbedder(model="text-embedding-3-large")
    """

    name = "openai"

    def __init__(
        self,
        model: str = CHUNK_EMBED_EMBEDDING_MODEL,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)


class CohereEmbedder(LiteLLMEmbedder):
    """
    Cohere embedder with smart defaults.

    Args:
        model: Cohere model (default: embed-english-v3.0)
        input_type: One of 'search_document', 'search_query', 'classification', 'clustering'

    Example:
        >>> embedder = CohereEmbedder()
        >>> embedder = CohereEmbedder(input_type="search_query")  # For queries
    """

    name = "cohere"

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        **kwargs: Any,
    ):
        # Add cohere/ prefix if not present
        if not model.startswith("cohere/"):
            model = f"cohere/{model}"

        super().__init__(model=model, input_type=input_type, **kwargs)


class VoyageEmbedder(LiteLLMEmbedder):
    """
    Voyage AI embedder (convenience alias).

    Example:
        >>> embedder = VoyageEmbedder()  # Uses voyage-01
    """

    name = "voyage"

    def __init__(
        self,
        model: str = "voyage-01",
        **kwargs: Any,
    ):
        if not model.startswith("voyage/"):
            model = f"voyage/{model}"
        super().__init__(model=model, **kwargs)
