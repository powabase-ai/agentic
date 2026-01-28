"""
Embedder - base class for embedding providers.

Embedders transform text into vector embeddings for semantic search.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Base class for embedding providers.

    An Embedder transforms text into dense vector embeddings suitable
    for semantic similarity search. Implementations use various model
    providers (OpenAI, Cohere, local models, etc.).

    Key Design Principles:
        1. **Sync and async**: Both embed() and aembed() methods
        2. **Batch support**: embed_batch() for efficiency
        3. **Model agnostic**: Works with any embedding model

    Subclasses:
        - OpenAIEmbedder: OpenAI models via LiteLLM

    Example:
        >>> class OpenAIEmbedder(Embedder):
        ...     def __init__(self, model="text-embedding-3-small"):
        ...         self.model = model
        ...
        ...     def embed(self, text: str) -> list[float]:
        ...         response = litellm.embedding(model=self.model, input=[text])
        ...         return response.data[0].embedding
    """

    # Subclasses should define this
    model: str = "unknown"
    dimensions: int | None = None  # Embedding dimensions (if known)

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats

        Example:
            >>> embedding = embedder.embed("Hello, world!")
            >>> len(embedding)
            1536
        """
        ...

    @abstractmethod
    async def aembed(self, text: str) -> list[float]:
        """
        Async version of embed().

        Args:
            text: The text to embed

        Returns:
            Embedding vector as a list of floats
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Default implementation calls embed() for each text.
        Subclasses can override for batch API calls.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Example:
            >>> embeddings = embedder.embed_batch(["Hello", "World"])
            >>> len(embeddings)
            2
        """
        return [self.embed(text) for text in texts]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Async version of embed_batch().

        Default implementation calls aembed() for each text.
        Subclasses can override for batch API calls.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Note: This is sequential; subclasses can implement parallel
        return [await self.aembed(text) for text in texts]

    def __repr__(self) -> str:
        dims = f", dims={self.dimensions}" if self.dimensions else ""
        return f"{self.__class__.__name__}(model={self.model!r}{dims})"
