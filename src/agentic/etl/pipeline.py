"""ETL pipeline orchestrator with fluent builder pattern"""

import asyncio
from typing import Optional
from .base import BaseConnector, BaseTransformer, BaseSink
from .registry import (
    ConnectorRegistry,
    ExtractorRegistry,
    ChunkerFactory,
    EmbedderFactory,
    SinkRegistry,
)


class Pipeline:
    """Fluent builder pattern for ETL pipeline configuration"""
    
    def __init__(self):
        self.connector: Optional[BaseConnector] = None
        self.extractor: Optional[BaseTransformer] = None
        self.chunker: Optional[BaseTransformer] = None
        self.embedder: Optional[BaseTransformer] = None
        self.sink: Optional[BaseSink] = None
    
    def source(self, source_type: str, **kwargs):
        """Set the source connector"""
        self.connector = ConnectorRegistry.create(source_type, kwargs)
        return self
    
    def extract(self, format: str = "auto", file_path: str = None, **kwargs):
        """Set the text extractor"""
        config = kwargs.copy()
        if file_path:
            config["file_path"] = file_path
        self.extractor = ExtractorRegistry.create(format, config)
        return self
    
    def chunk(self, strategy: str = "recursive", size: int = 1000, overlap: int = 200, **kwargs):
        """Set the chunking strategy"""
        self.chunker = ChunkerFactory.create(
            strategy, chunk_size=size, chunk_overlap=overlap, **kwargs
        )
        return self
    
    def embed(self, model: str = "text-embedding-3-small", provider: str = "openai", **kwargs):
        """Set the embedding provider"""
        self.embedder = EmbedderFactory.create(provider, model=model, **kwargs)
        return self
    
    def load(self, sink_type: str, **kwargs):
        """Set the storage sink"""
        self.sink = SinkRegistry.create(sink_type, kwargs)
        return self
    
    def process(self, file_path: str):
        """Execute the pipeline"""
        # 1. Load document from source (if connector specified)
        if self.connector:
            file_path = self.connector.load(file_path)
        
        # 2. Extract text
        if self.extractor:
            # Extractors have an extract() method that takes file_path
            if hasattr(self.extractor, 'extract'):
                result = self.extractor.extract(file_path)
                extraction = {"text": result.text, "metadata": result.metadata, "file_path": file_path}
            else:
                extraction = self.extractor.transform(file_path)
                extraction["file_path"] = file_path
        else:
            # Fallback: simple file read
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    extraction = {"text": f.read(), "metadata": {}}
            except UnicodeDecodeError:
                # Binary file - can't read as text
                raise ValueError(f"Cannot extract text from binary file: {file_path}. Please use an extractor.")
        
        # 3. Chunk text
        if self.chunker:
            chunks = self.chunker.transform(extraction)
        else:
            # Fallback: single chunk
            chunks = [{"text": extraction["text"], "start_char": 0, "end_char": len(extraction["text"]), "metadata": {}}]
        
        # 4. Generate embeddings
        if self.embedder:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = self.embedder.transform(chunk_texts)
            # Attach embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i] if i < len(embeddings) else None
        else:
            embeddings = []
        
        # 5. Store results
        if self.sink:
            return self.sink.store(extraction, chunks)
        else:
            return {"extraction": extraction, "chunks": chunks, "embeddings": embeddings}
    
    async def run(self, file_path: str):
        """Async execution"""
        return await asyncio.to_thread(self.process, file_path)

