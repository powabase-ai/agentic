"""LiteLLM embedder - unified interface for multiple embedding providers

Supports all providers from litellm:
- OpenAI: text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002
- Cohere: embed-english-v3.0, embed-multilingual-v3.0
- Vertex AI: textembedding-gecko, text-embedding-004
- Bedrock: amazon.titan-embed-text-v1, cohere.embed-english-v3
- Mistral: mistral-embed
- Voyage: voyage-01, voyage-lite-01
- HuggingFace: any model via huggingface/ prefix
- And many more...

Usage examples:
    # OpenAI
    embedder = LiteLLMEmbedder(model="text-embedding-3-small")

    # Cohere with custom params
    embedder = LiteLLMEmbedder(
        model="cohere/embed-english-v3.0",
        input_type="search_document"
    )

    # Vertex AI
    embedder = LiteLLMEmbedder(model="vertex_ai/textembedding-gecko")

    # Local HuggingFace model
    embedder = LiteLLMEmbedder(model="huggingface/sentence-transformers/all-MiniLM-L6-v2")
"""

import logging
from typing import List, Optional, Any, Dict

from litellm import embedding as litellm_embedding

from .base import BaseEmbedder
from ...registry import EmbedderFactory

logger = logging.getLogger(__name__)


@EmbedderFactory.register("litellm")
class LiteLLMEmbedder(BaseEmbedder):
    """Universal embedder using LiteLLM for multi-provider support

    Args:
        model: Model identifier (e.g., 'text-embedding-3-small', 'cohere/embed-english-v3.0')
        api_key: Optional API key (defaults to environment variable)
        api_base: Optional custom API endpoint
        dimensions: Optional embedding dimensions (supported by some models)
        encoding_format: Output format - "float" (default) or "base64"
        timeout: Request timeout in seconds (default: 600)
        **provider_params: Provider-specific parameters (e.g., input_type for Cohere)
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        dimensions: Optional[int] = None,
        encoding_format: str = "float",
        timeout: int = 600,
        **provider_params: Any,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self.timeout = timeout
        self.provider_params = provider_params

        logger.info(f"Initialized LiteLLM embedder with model: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        if not texts:
            return []

        # Build kwargs for litellm
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "encoding_format": self.encoding_format,
            "timeout": self.timeout,
        }

        # Add optional parameters if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        # Add provider-specific params
        kwargs.update(self.provider_params)

        try:
            # LiteLLM handles batching, retries, and rate limiting automatically
            response = litellm_embedding(**kwargs)

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in response["data"]]

            logger.debug(
                f"Generated {len(embeddings)} embeddings using {self.model} "
                f"(tokens: {response.get('usage', {}).get('total_tokens', 'unknown')})"
            )

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed with {self.model}: {e}")
            raise


# Convenience aliases for common providers
@EmbedderFactory.register("openai")
class OpenAIEmbedder(LiteLLMEmbedder):
    """OpenAI embedder (alias for LiteLLM with OpenAI defaults)"""

    def __init__(self, model: str = "text-embedding-3-small", **kwargs):
        super().__init__(model=model, **kwargs)


@EmbedderFactory.register("cohere")
class CohereEmbedder(LiteLLMEmbedder):
    """Cohere embedder with smart defaults

    Args:
        model: Cohere model (default: embed-english-v3.0)
        input_type: One of 'search_document', 'search_query', 'classification', 'clustering'
                   (default: 'search_document')
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        **kwargs,
    ):
        # Add cohere/ prefix if not present
        if not model.startswith("cohere/"):
            model = f"cohere/{model}"

        super().__init__(model=model, input_type=input_type, **kwargs)


@EmbedderFactory.register("voyage")
class VoyageEmbedder(LiteLLMEmbedder):
    """Voyage AI embedder (alias for LiteLLM with Voyage defaults)"""

    def __init__(self, model: str = "voyage-01", **kwargs):
        if not model.startswith("voyage/"):
            model = f"voyage/{model}"
        super().__init__(model=model, **kwargs)


@EmbedderFactory.register("sentence-transformers")
class SentenceTransformersEmbedder(LiteLLMEmbedder):
    """Local sentence transformers via LiteLLM/HuggingFace"""

    def __init__(self, model: str = "all-MiniLM-L6-v2", **kwargs):
        # Use HuggingFace inference API via LiteLLM
        if not model.startswith("huggingface/"):
            model = f"huggingface/sentence-transformers/{model}"
        super().__init__(model=model, **kwargs)
