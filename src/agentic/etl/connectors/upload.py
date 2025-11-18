"""Upload connector for handling direct API uploads"""

import os
import tempfile
from .base import BaseConnector
from ..registry import ConnectorRegistry


@ConnectorRegistry.register("upload")
class UploadConnector(BaseConnector):
    """Connector for handling uploaded files (already saved locally)"""
    
    def __init__(self, upload_dir: str = None, **kwargs):
        self.upload_dir = upload_dir or tempfile.gettempdir()
    
    def load(self, source: str) -> str:
        """Load uploaded file (source is already a local path)"""
        # If source is relative, check in upload_dir
        if not os.path.isabs(source):
            full_path = os.path.join(self.upload_dir, source)
            if os.path.exists(full_path):
                return full_path
        
        # Otherwise treat as absolute path
        if os.path.exists(source):
            return source
        
        raise FileNotFoundError(f"Uploaded file not found: {source}")

