"""
FileUploadConnector - simple connector for direct file uploads.

This connector handles content that's already available as bytes
(e.g., from HTTP file uploads, local file reads).
"""

import mimetypes
from datetime import datetime
from pathlib import Path

from agentic.ingest.connector.base import Connector
from agentic.ingest.models import ContentItem, RawContent


class FileUploadConnector(Connector):
    """
    Connector for direct file uploads and local files.

    This is the simplest connector - it takes bytes that are already
    available and wraps them in RawContent. Useful for:
    - HTTP file uploads (bytes from request)
    - Reading local files
    - Testing with in-memory content

    Example:
        >>> connector = FileUploadConnector()
        >>>
        >>> # From bytes (e.g., HTTP upload)
        >>> raw = await connector.fetch_bytes(
        ...     content=uploaded_bytes,
        ...     filename="report.pdf",
        ...     mime_type="application/pdf",
        ... )
        >>>
        >>> # From local file
        >>> raw = await connector.fetch_file("/path/to/report.pdf")
    """

    name = "file_upload"

    async def fetch(self, item: ContentItem) -> RawContent:
        """
        Fetch content from a ContentItem.

        For file uploads, the ContentItem should have been created with
        the content stored in metadata["content"].

        Args:
            item: ContentItem with content in metadata

        Returns:
            RawContent with the bytes and metadata
        """
        content = item.metadata.get("content")
        if content is None:
            raise ValueError(
                f"ContentItem {item.uri} has no content in metadata. "
                "Use fetch_bytes() or fetch_file() instead."
            )

        return RawContent(
            content=content,
            mime_type=item.mime_type or "application/octet-stream",
            source_uri=item.uri,
            filename=item.name,
            metadata=item.metadata,
        )

    async def fetch_bytes(
        self,
        content: bytes,
        filename: str,
        mime_type: str | None = None,
        metadata: dict | None = None,
    ) -> RawContent:
        """
        Create RawContent directly from bytes.

        This is the most common method for handling HTTP file uploads.

        Args:
            content: The raw bytes
            filename: Original filename
            mime_type: MIME type (will guess from filename if not provided)
            metadata: Additional metadata to attach

        Returns:
            RawContent ready for extraction

        Example:
            >>> # From Flask/FastAPI file upload
            >>> raw = await connector.fetch_bytes(
            ...     content=request.files["file"].read(),
            ...     filename=request.files["file"].filename,
            ...     mime_type=request.files["file"].content_type,
            ... )
        """
        # Guess MIME type if not provided
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(filename)
            mime_type = guessed or "application/octet-stream"

        return RawContent(
            content=content,
            mime_type=mime_type,
            source_uri=f"upload://{filename}",
            filename=filename,
            metadata=metadata or {},
            fetched_at=datetime.now(),
        )

    async def fetch_file(
        self,
        file_path: str | Path,
        mime_type: str | None = None,
        metadata: dict | None = None,
    ) -> RawContent:
        """
        Read a local file and create RawContent.

        Args:
            file_path: Path to the file
            mime_type: MIME type (will guess from extension if not provided)
            metadata: Additional metadata to attach

        Returns:
            RawContent with file contents

        Example:
            >>> raw = await connector.fetch_file("/path/to/document.pdf")
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_bytes()

        # Guess MIME type if not provided
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(str(path))
            mime_type = guessed or "application/octet-stream"

        return RawContent(
            content=content,
            mime_type=mime_type,
            source_uri=f"file://{path.absolute()}",
            filename=path.name,
            size_bytes=len(content),
            metadata={
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                **(metadata or {}),
            },
            fetched_at=datetime.now(),
        )
