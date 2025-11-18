"""Image extractor using vision models via LiteLLM"""

import base64
import logging
from typing import Optional

from litellm import completion

from .base import BaseExtractor, ExtractionResult
from ...registry import ExtractorRegistry

logger = logging.getLogger(__name__)

# Supported image formats
IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]

# MIME type mapping
EXTENSION_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


@ExtractorRegistry.register("image")
class ImageExtractor(BaseExtractor):
    """Extract text from images using vision models via LiteLLM

    Supports any vision-capable model through LiteLLM, including:
    - OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4-vision-preview
    - Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
    - Google: gemini-pro-vision, gemini-1.5-pro
    - And many more: https://docs.litellm.ai/docs/providers
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """Initialize ImageExtractor

        Args:
            model: Model name (e.g., "gpt-4o", "claude-3-sonnet-20240229", "gemini-pro-vision")
            api_key: API key for the provider (or set via environment variables)
            api_base: Custom API base URL (optional)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            prompt_template: Custom prompt template (must include {filename} placeholder)
            **kwargs: Additional parameters passed to litellm.completion()
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_params = kwargs

        # Default prompt template
        self.prompt_template = prompt_template or (
            'Based on the provided image with filename "{filename}", respond with the following in JSON format:\n'
            "title: An appropriate title for the image.\n"
            'short_description: A 1-sentence description of the image. Start the description with "a photo of...", "a drawing of...", "a screenshot of...", etc.\n'
            "long_description: A detailed description of the image content, including any visible text, objects, people, settings, and other relevant details."
        )

    def _get_mime_type(self, file_path: str) -> str:
        """Get MIME type from file extension"""
        ext = file_path.split(".")[-1].lower()
        return EXTENSION_TO_MIME.get(ext, "image/jpeg")

    def extract(self, file_path: str) -> ExtractionResult:
        """Extract text description from image using vision model

        Args:
            file_path: Path to the image file

        Returns:
            ExtractionResult with extracted description and metadata
        """
        try:
            # Read and encode image
            with open(file_path, "rb") as f:
                image_content = f.read()

            b64_content = base64.b64encode(image_content).decode("utf-8")
            mime_type = self._get_mime_type(file_path)

            # Get filename for prompt
            import os

            filename = os.path.basename(file_path)

            # Prepare messages with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.prompt_template.format(filename=filename),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_content}"
                            },
                        },
                    ],
                }
            ]

            # Prepare completion parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                **self.extra_params,
            }

            # Add API key if provided
            if self.api_key:
                completion_params["api_key"] = self.api_key

            # Add API base if provided
            if self.api_base:
                completion_params["api_base"] = self.api_base

            logger.info(f"Calling vision model {self.model} for image: {filename}")

            # Call LiteLLM
            response = completion(**completion_params)

            # Parse response
            import json

            result = json.loads(response.choices[0].message.content)

            # Extract fields
            title = result.get("title", "")
            short_description = result.get("short_description", "")
            long_description = result.get("long_description", "")

            # Use long_description as primary text (like legacy code)
            text = long_description or short_description or title

            if not text:
                raise ValueError("Vision model returned empty description")

            # Build metadata
            metadata = {
                "extraction_method": "vision_model",
                "model": self.model,
                "title": title,
                "short_description": short_description,
                "long_description": long_description,
                "filename": filename,
                "mime_type": mime_type,
                "file_size": len(image_content),
            }

            # Add usage info if available
            if hasattr(response, "usage") and response.usage:
                metadata["tokens"] = response.usage.total_tokens
                metadata["prompt_tokens"] = response.usage.prompt_tokens
                metadata["completion_tokens"] = response.usage.completion_tokens

            logger.info(
                f"Successfully extracted text from image: {filename} ({len(text)} chars)"
            )

            return ExtractionResult(text=text, metadata=metadata)

        except Exception as e:
            logger.error(
                f"Failed to extract from image {file_path}: {e}", exc_info=True
            )
            raise RuntimeError(f"Image extraction failed: {e}")
