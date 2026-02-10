"""
Retrieval module - algorithms for querying indexed content.

Retrieval algorithms query the knowledge store and post-process results
to return relevant context for agents.

Built-in algorithms:
- VectorSearchAlgorithm: Semantic similarity search using embeddings
- HybridSearchAlgorithm: Combines vector + keyword search (placeholder)

Future algorithms:
- graph_traversal: Navigate entity relationships (GraphRAG)

Example:
    >>> from agentic.knowledge.retrieval import VectorSearchAlgorithm
    >>> from agentic.knowledge.embedder import OpenAIEmbedder
    >>>
    >>> retriever = VectorSearchAlgorithm(embedder=OpenAIEmbedder())
    >>> chunks = await retriever.aretrieve(
    ...     query="How do I reset my password?",
    ...     store=knowledge_store,
    ...     config=RetrievalConfig(top_k=5),
    ... )
"""

from agentic.knowledge.retrieval.base import RetrievalAlgorithm
from agentic.knowledge.retrieval.tree_search import TreeSearchAlgorithm
from agentic.knowledge.retrieval.vector_search import (
    HybridSearchAlgorithm,
    VectorSearchAlgorithm,
)

__all__ = [
    "RetrievalAlgorithm",
    "VectorSearchAlgorithm",
    "HybridSearchAlgorithm",
    "TreeSearchAlgorithm",
]
