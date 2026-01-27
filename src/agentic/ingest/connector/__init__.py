"""
Connector module - how content enters the system.

Connectors provide a unified interface for fetching content from various
sources: file uploads, cloud storage, web pages, APIs, etc.

Built-in connectors:
- FileUploadConnector: For direct file uploads (bytes passed in)

Future connectors (not yet implemented):
- S3Connector: AWS S3 bucket
- GCSConnector: Google Cloud Storage
- WebCrawlerConnector: Fetch web pages
- WebhookConnector: Receive content via webhooks

Example:
    >>> from agentic.ingest.connector import FileUploadConnector
    >>>
    >>> connector = FileUploadConnector()
    >>> raw = await connector.fetch_bytes(
    ...     content=file_bytes,
    ...     filename="report.pdf",
    ...     mime_type="application/pdf",
    ... )
"""

from agentic.ingest.connector.base import Connector
from agentic.ingest.connector.file_upload import FileUploadConnector

__all__ = [
    "Connector",
    "FileUploadConnector",
]
