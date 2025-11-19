"""File system connector"""

import os
from .base import BaseConnector
from ..registry import ConnectorRegistry


@ConnectorRegistry.register("file")
class FileConnector(BaseConnector):
    """Connector for local file system"""
    
    def __init__(self, **kwargs):
        pass
    
    def load(self, source: str) -> str:
        """Load file from local filesystem"""
        if not os.path.exists(source):
            raise FileNotFoundError(f"File not found: {source}")
        return source

