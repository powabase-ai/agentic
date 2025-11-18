"""PostgreSQL sink with pgvector support"""

from typing import Any, Dict, List, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from pgvector.sqlalchemy import Vector

from ..base import BaseSink
from ..registry import SinkRegistry


@SinkRegistry.register("postgresql")
class PostgreSQLSink(BaseSink):
    """PostgreSQL storage sink with pgvector for embeddings"""
    
    def __init__(
        self,
        connection_string: str,
        document_table: str = "documents",
        chunk_table: str = "chunks",
        **kwargs
    ):
        self.connection_string = connection_string
        self.document_table = document_table
        self.chunk_table = chunk_table
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    def _ensure_pgvector_extension(self, session: Session):
        """Ensure pgvector extension is installed"""
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        session.commit()
    
    def store(self, extraction: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store extraction and chunks in PostgreSQL"""
        session = self.Session()
        
        try:
            # Ensure pgvector extension
            self._ensure_pgvector_extension(session)
            
            # Store document
            doc_data = {
                "raw_content": extraction["text"],
                "file_path": extraction.get("file_path", ""),
                "file_size": extraction.get("file_size", 0),
                "file_type": extraction.get("file_type", ""),
                "extraction_method": extraction.get("metadata", {}).get("extraction_method"),
                "meta": extraction.get("metadata", {}),
            }
            
            # Insert document (assuming table exists with proper schema)
            # This is a simplified version - actual implementation would use ORM models
            doc_insert = text(f"""
                INSERT INTO {self.document_table} 
                (raw_content, file_path, file_size, file_type, extraction_method, meta)
                VALUES (:raw_content, :file_path, :file_size, :file_type, :extraction_method, :meta)
                RETURNING id
            """)
            result = session.execute(doc_insert, doc_data)
            document_id = result.scalar()
            
            # Store chunks with embeddings
            chunk_data_list = []
            for chunk in chunks:
                embedding = chunk.get("embedding")
                chunk_data = {
                    "document_id": document_id,
                    "text": chunk["text"],
                    "embedding": embedding if embedding else None,
                    "start_char": chunk["start_char"],
                    "end_char": chunk["end_char"],
                    "chunk_index": chunk.get("metadata", {}).get("chunk_index", 0),
                    "meta": chunk.get("metadata", {}),
                }
                chunk_data_list.append(chunk_data)
            
            # Insert chunks
            if chunk_data_list:
                chunk_insert = text(f"""
                    INSERT INTO {self.chunk_table}
                    (document_id, text, embedding, start_char, end_char, chunk_index, meta)
                    VALUES (:document_id, :text, :embedding::vector, :start_char, :end_char, :chunk_index, :meta)
                """)
                session.execute(chunk_insert, chunk_data_list)
            
            session.commit()
            
            return {
                "document_id": document_id,
                "chunks_stored": len(chunk_data_list),
                "status": "success"
            }
            
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

