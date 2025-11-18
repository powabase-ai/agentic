"""S3 connector"""

import os
import tempfile
import boto3
from botocore.exceptions import ClientError
from .base import BaseConnector
from ..registry import ConnectorRegistry


@ConnectorRegistry.register("s3")
class S3Connector(BaseConnector):
    """Connector for AWS S3"""
    
    def __init__(self, bucket: str, aws_access_key_id: str = None, aws_secret_access_key: str = None, region_name: str = "us-east-1", **kwargs):
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )
    
    def load(self, source: str) -> str:
        """Download file from S3 and return local path"""
        # Extract key from S3 path (s3://bucket/key or just key)
        if source.startswith("s3://"):
            parts = source[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.bucket
            key = source
        
        # Create temp file
        _, ext = os.path.splitext(key)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            self.s3_client.download_file(bucket, key, temp_path)
            return temp_path
        except ClientError as e:
            os.unlink(temp_path)
            raise FileNotFoundError(f"Failed to download from S3: {e}")

