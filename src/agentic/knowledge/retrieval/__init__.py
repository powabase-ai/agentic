"""
Retrieval module - algorithms for querying indexed content.

Retrieval algorithms query the knowledge store and post-process results
to return relevant context for agents.

Built-in algorithms:
- VectorSearchAlgorithm: Semantic similarity search using embeddings
- FullTextSearchAlgorithm: BM25 keyword search
- HybridSearchAlgorithm: Combines vector + keyword search via RRF

Utilities:
- reciprocal_rank_fusion: Rank-based result fusion for combining retrieval signals

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
from agentic.knowledge.retrieval.bm25 import bm25_score, parse_tsvector
from agentic.knowledge.retrieval.fusion import reciprocal_rank_fusion
from agentic.knowledge.retrieval.tree_search import SelectedNode, TreeSearchAlgorithm
from agentic.knowledge.retrieval.vector_search import (
    FullTextSearchAlgorithm,
    HybridSearchAlgorithm,
    VectorSearchAlgorithm,
)

__all__ = [
    "RetrievalAlgorithm",
    "VectorSearchAlgorithm",
    "FullTextSearchAlgorithm",
    "HybridSearchAlgorithm",
    "TreeSearchAlgorithm",
    "SelectedNode",
    "reciprocal_rank_fusion",
    "bm25_score",
    "parse_tsvector",
]
