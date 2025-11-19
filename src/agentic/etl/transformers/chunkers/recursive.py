"""Recursive text chunker"""

from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker, Chunk
from ...registry import ChunkerFactory


@ChunkerFactory.register("recursive")
class RecursiveChunker(BaseChunker):
    """Recursive character text splitter"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """Split text into chunks using recursive splitting"""
        # Use LangChain's splitter
        documents = self.splitter.create_documents([text])
        
        chunks = []
        current_pos = 0
        
        for i, doc in enumerate(documents):
            chunk_text = doc.page_content
            start_char = text.find(chunk_text, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_text)
            current_pos = end_char
            
            chunk_metadata = doc.metadata.copy()
            if metadata:
                chunk_metadata.update(metadata)
            chunk_metadata["chunk_index"] = i
            
            chunks.append(Chunk(
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata,
            ))
        
        return chunks

