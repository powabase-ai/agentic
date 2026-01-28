"""
Tests for the Knowledge module structure and base classes.
"""

import pytest

from agentic.knowledge import (
    ChunkingStrategy,
    Embedder,
    IndexingAlgorithm,
    IndexingConfig,
    IndexResult,
    KnowledgeStore,
    RetrievalAlgorithm,
    RetrievalConfig,
    RetrievedChunk,
)
from agentic.knowledge.chunking import TextChunk


class TestModels:
    """Tests for knowledge module data models."""

    def test_indexing_config_defaults(self):
        """IndexingConfig should have sensible defaults."""
        config = IndexingConfig()

        assert config.strategy == "chunk_embed"
        assert config.chunk_size == 500
        assert config.overlap == 50
        assert config.embedding_model == "text-embedding-3-small"
        assert config.extra == {}

    def test_indexing_config_custom(self):
        """IndexingConfig should accept custom values."""
        config = IndexingConfig(
            strategy="graph",
            chunk_size=1000,
            overlap=100,
            embedding_model="text-embedding-3-large",
            extra={"custom_key": "value"},
        )

        assert config.strategy == "graph"
        assert config.chunk_size == 1000
        assert config.extra["custom_key"] == "value"

    def test_retrieval_config_defaults(self):
        """RetrievalConfig should have sensible defaults."""
        config = RetrievalConfig()

        assert config.method == "vector_search"
        assert config.top_k == 5
        assert config.similarity_threshold == 0.0
        assert config.extra == {}

    def test_index_result(self):
        """IndexResult should store basic indexing metadata."""
        result = IndexResult(
            strategy_name="chunk_embed",
            source_id="source-123",
            stats={"chunk_count": 10},
        )

        assert result.strategy_name == "chunk_embed"
        assert result.source_id == "source-123"
        assert result.stats["chunk_count"] == 10
        assert result.indexed_at is not None

    def test_retrieved_chunk(self):
        """RetrievedChunk should represent a search result."""
        chunk = RetrievedChunk(
            chunk_id="chunk-123",
            text="This is the chunk content",
            score=0.95,
            source_id="source-456",
            knowledge_base_id="kb-789",
            meta={"page": 5},
        )

        assert chunk.chunk_id == "chunk-123"
        assert chunk.text == "This is the chunk content"
        assert chunk.score == 0.95
        assert chunk.meta["page"] == 5

    def test_retrieved_chunk_repr(self):
        """RetrievedChunk should have a useful repr."""
        chunk = RetrievedChunk(
            chunk_id="chunk-123",
            text="Short text",
            score=0.85,
        )

        repr_str = repr(chunk)
        assert "0.85" in repr_str
        assert "Short text" in repr_str

    def test_text_chunk(self):
        """TextChunk should represent a chunk from chunking."""
        chunk = TextChunk(
            text="Chunk content here",
            index=0,
            start_char=0,
            end_char=18,
            metadata={"section": "intro"},
        )

        assert chunk.text == "Chunk content here"
        assert chunk.index == 0
        assert len(chunk) == 18
        assert chunk.metadata["section"] == "intro"


class TestChunkingStrategy:
    """Tests for ChunkingStrategy base class."""

    def test_chunking_strategy_init(self):
        """ChunkingStrategy should validate init parameters."""

        # Create a concrete subclass for testing
        class TestChunker(ChunkingStrategy):
            name = "test"

            def chunk(self, text, source_id=None):
                return []

        chunker = TestChunker(chunk_size=500, overlap=50)
        assert chunker.chunk_size == 500
        assert chunker.overlap == 50

    def test_chunking_strategy_validation(self):
        """ChunkingStrategy should reject invalid parameters."""

        class TestChunker(ChunkingStrategy):
            name = "test"

            def chunk(self, text, source_id=None):
                return []

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TestChunker(chunk_size=0)

        with pytest.raises(ValueError, match="overlap must be non-negative"):
            TestChunker(chunk_size=100, overlap=-1)

        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            TestChunker(chunk_size=100, overlap=100)

    def test_chunking_strategy_repr(self):
        """ChunkingStrategy should have a useful repr."""

        class TestChunker(ChunkingStrategy):
            name = "test"

            def chunk(self, text, source_id=None):
                return []

        chunker = TestChunker(chunk_size=500, overlap=50)
        repr_str = repr(chunker)
        assert "TestChunker" in repr_str
        assert "500" in repr_str
        assert "50" in repr_str


class TestEmbedder:
    """Tests for Embedder base class."""

    def test_embedder_batch_default(self):
        """Embedder.embed_batch should call embed() for each text."""

        class TestEmbedder(Embedder):
            model = "test-model"

            def embed(self, text):
                return [len(text)]  # Simple mock: return length as embedding

            async def aembed(self, text):
                return [len(text)]

        embedder = TestEmbedder()
        embeddings = embedder.embed_batch(["hello", "world!"])

        assert embeddings == [[5], [6]]

    def test_embedder_repr(self):
        """Embedder should have a useful repr."""

        class TestEmbedder(Embedder):
            model = "test-model"
            dimensions = 384

            def embed(self, text):
                return []

            async def aembed(self, text):
                return []

        embedder = TestEmbedder()
        repr_str = repr(embedder)
        assert "TestEmbedder" in repr_str
        assert "test-model" in repr_str
        assert "384" in repr_str


class TestIndexingAlgorithm:
    """Tests for IndexingAlgorithm base class."""

    def test_indexing_algorithm_repr(self):
        """IndexingAlgorithm should have a useful repr."""

        class TestIndexer(IndexingAlgorithm):
            name = "test_indexer"

            def index(self, content, config):
                return IndexResult(strategy_name=self.name)

        indexer = TestIndexer()
        repr_str = repr(indexer)
        assert "TestIndexer" in repr_str
        assert "test_indexer" in repr_str

    @pytest.mark.asyncio
    async def test_indexing_algorithm_aindex_default(self):
        """IndexingAlgorithm.aindex should call sync index by default."""

        class TestIndexer(IndexingAlgorithm):
            name = "test_indexer"

            def index(self, content, config):
                return IndexResult(
                    strategy_name=self.name,
                    stats={"content_length": len(content)},
                )

        indexer = TestIndexer()
        config = IndexingConfig()
        result = await indexer.aindex("test content", config)

        assert result.strategy_name == "test_indexer"
        assert result.stats["content_length"] == 12


class TestRetrievalAlgorithm:
    """Tests for RetrievalAlgorithm base class."""

    def test_retrieval_algorithm_repr(self):
        """RetrievalAlgorithm should have a useful repr."""

        class TestRetriever(RetrievalAlgorithm):
            name = "test_retriever"

            async def retrieve(self, query, store, config):
                return []

        retriever = TestRetriever()
        repr_str = repr(retriever)
        assert "TestRetriever" in repr_str
        assert "test_retriever" in repr_str

    def test_apply_threshold(self):
        """RetrievalAlgorithm._apply_threshold should filter by score."""

        class TestRetriever(RetrievalAlgorithm):
            name = "test_retriever"

            async def retrieve(self, query, store, config):
                return []

        retriever = TestRetriever()

        chunks = [
            RetrievedChunk(chunk_id="1", text="a", score=0.9),
            RetrievedChunk(chunk_id="2", text="b", score=0.5),
            RetrievedChunk(chunk_id="3", text="c", score=0.3),
        ]

        # No filtering with 0 threshold
        filtered = retriever._apply_threshold(chunks, 0.0)
        assert len(filtered) == 3

        # Filter below 0.5
        filtered = retriever._apply_threshold(chunks, 0.5)
        assert len(filtered) == 2
        assert all(c.score >= 0.5 for c in filtered)

        # Filter below 0.8
        filtered = retriever._apply_threshold(chunks, 0.8)
        assert len(filtered) == 1
        assert filtered[0].score == 0.9


class TestKnowledgeStore:
    """Tests for KnowledgeStore abstract interface."""

    def test_knowledge_store_is_abstract(self):
        """KnowledgeStore should not be instantiable directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            KnowledgeStore()

    def test_knowledge_store_subclass(self):
        """KnowledgeStore subclass must implement all methods."""

        class PartialStore(KnowledgeStore):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            PartialStore()


class TestModuleImports:
    """Tests that module exports are correct."""

    def test_knowledge_module_exports(self):
        """Main knowledge module should export all base classes."""
        from agentic import knowledge

        # Check base classes
        assert hasattr(knowledge, "KnowledgeStore")
        assert hasattr(knowledge, "IndexingAlgorithm")
        assert hasattr(knowledge, "RetrievalAlgorithm")
        assert hasattr(knowledge, "ChunkingStrategy")
        assert hasattr(knowledge, "Embedder")

        # Check models
        assert hasattr(knowledge, "IndexResult")
        assert hasattr(knowledge, "RetrievedChunk")
        assert hasattr(knowledge, "IndexingConfig")
        assert hasattr(knowledge, "RetrievalConfig")
