"""
ChunkingStrategy - base class for text chunking algorithms.

Chunking strategies split text into smaller pieces suitable for
embedding and retrieval.
"""

from abc import ABC, abstractmethod

from agentic.knowledge.models import TextChunk


class ChunkingStrategy(ABC):
    """
    Base class for text chunking strategies.

    A ChunkingStrategy splits text into smaller chunks suitable for
    embedding and retrieval. Different strategies optimize for different
    use cases (fixed size, semantic boundaries, etc.).

    Key Design Principles:
        1. **Pure functions**: chunk() returns data, no side effects
        2. **Configurable**: Chunk size and overlap controlled via init
        3. **Metadata preservation**: Can pass through document metadata

    Subclasses:
        - RecursiveChunking: Split by separators (paragraph → sentence → word)
        - FixedSizeChunking: Fixed character count with overlap
        - SemanticChunking: Split by semantic boundaries (future)

    Example:
        >>> class RecursiveChunking(ChunkingStrategy):
        ...     name = "recursive"
        ...     separators = ["\\n\\n", "\\n", ". ", " "]
        ...
        ...     def chunk(self, text: str) -> list[TextChunk]:
        ...         # Split recursively by separators
        ...         ...
    """

    # Subclasses must define this
    name: str = "base"

    def __init__(
        self,
        chunk_size: int = 2000,
        overlap: int = 50,
    ) -> None:
        """
        Initialize the chunking strategy.

        Args:
            chunk_size: Target size for chunks (in tokens)
            overlap: Overlap between consecutive chunks (in tokens)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(
        self,
        text: str,
        source_id: str | None = None,
    ) -> list[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: The text to split
            source_id: Optional identifier of the source document

        Returns:
            List of TextChunk objects

        Example:
            >>> chunks = chunker.chunk("Long document text...")
            >>> len(chunks)
            10
            >>> chunks[0].text
            "Long document..."
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.overlap})"
