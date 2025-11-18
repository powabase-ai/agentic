"""Base database models and mixins for ETL pipeline"""

from datetime import datetime
from typing import Optional

from sqlalchemy import func, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class DocumentMixin:
    """Mixin for document models"""

    raw_content: Mapped[str]  # Full extracted text
    file_path: Mapped[str]
    file_size: Mapped[int]
    file_type: Mapped[str]  # pdf, docx, txt, etc.
    extraction_method: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )  # Which extractor succeeded
    tokens: Mapped[Optional[int]] = mapped_column(nullable=True)  # Total token count
    meta: Mapped[dict] = mapped_column(JSONB, default={})  # Flexible metadata
    status: Mapped[str] = mapped_column(String, default="pending")  # ETL status
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ChunkMixin:
    """Mixin for chunk models"""

    text: Mapped[str]
    embedding: Mapped[Optional[Vector]] = mapped_column(
        Vector(1536), nullable=True
    )  # pgvector
    start_char: Mapped[int]
    end_char: Mapped[int]
    chunk_index: Mapped[int]
    tokens: Mapped[Optional[int]] = mapped_column(nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default={})  # page, headers, etc.


class ETLJobMixin:
    """Mixin for ETL job tracking"""

    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    progress: Mapped[int] = mapped_column(default=0)  # 0-100
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
