"""
Doc2JSON Indexing Algorithm - structured JSON extraction via sliding window.

This algorithm scans documents using a sliding window, iteratively extracting
structured data into a user-defined JSON schema. Each window produces a summary
and updates the accumulated JSON state.

Key features:
- Sliding window with configurable size and overlap
- User-defined JSON schema with nested objects/arrays
- Last-value-wins conflict resolution for scalars
- Append-all merge strategy for arrays
- Combined summary + extraction in single LLM call for efficiency
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import supports_response_schema

from agentic.knowledge.chunking.fixed import FixedSizeChunking
from agentic.knowledge.embedder import LiteLLMEmbedder
from agentic.knowledge.indexing.base import IndexingAlgorithm
from agentic.knowledge.model_config import (
    DOC2JSON_DEFAULT_PAGES_PER_WINDOW,
    DOC2JSON_DEFAULT_WINDOW_OVERLAP,
    DOC2JSON_DEFAULT_WINDOW_SIZE,
    DOC2JSON_EMBEDDING_MODEL,
    DOC2JSON_EXTRACTION_MAX_TOKENS,
    DOC2JSON_EXTRACTION_MODEL,
    DOC2JSON_MAX_RETRIES,
    DOC2JSON_SUMMARY_MAX_TOKENS,
    DOC2JSON_USE_IMAGES,
)
from agentic.knowledge.models import IndexingConfig

logger = logging.getLogger(__name__)


@dataclass
class Doc2JSONResult:
    """Result of doc2json indexing for a single document."""

    extracted_json: dict
    """The final accumulated JSON object conforming to the user schema."""

    combined_summary: str
    """Combined summary of the entire document for embedding/search."""

    combined_summary_embedding: list[float]
    """Embedding vector for the combined summary."""

    window_summaries: list[dict]
    """Per-window summaries with extraction metadata for debugging."""

    json_schema: dict
    """The user-defined JSON schema (snapshot)."""

    stats: dict = field(default_factory=dict)
    """Statistics: window_count, input_tokens, summary_tokens, etc."""


class Doc2JSONAlgorithm(IndexingAlgorithm):
    """
    Indexing algorithm for structured JSON extraction.

    Processes documents by:
    1. Splitting into overlapping windows using FixedSizeChunking
    2. For each window: LLM extracts summary + updates JSON fields
    3. Merging extractions (last-wins for scalars, append for arrays)
    4. Generating a combined summary from all window summaries
    5. Embedding the combined summary for retrieval

    The user defines a JSON schema specifying field names, types,
    descriptions, defaults, and example values.

    Example schema:
        {
            "fields": [
                {
                    "name": "company_name",
                    "type": "string",
                    "description": "Full legal name of the company",
                    "default": null,
                    "examples": ["Acme Corp", "TechStart Inc"]
                },
                {
                    "name": "products",
                    "type": "array",
                    "item_type": "string",
                    "description": "List of products offered",
                    "default": []
                }
            ]
        }
    """

    name = "doc2json"

    def __init__(
        self,
        model: str = DOC2JSON_EXTRACTION_MODEL,
        embedding_model: str = DOC2JSON_EMBEDDING_MODEL,
        window_size: int = DOC2JSON_DEFAULT_WINDOW_SIZE,
        window_overlap: int = DOC2JSON_DEFAULT_WINDOW_OVERLAP,
        use_images: bool = DOC2JSON_USE_IMAGES,
        pages_per_window: int = DOC2JSON_DEFAULT_PAGES_PER_WINDOW,
    ):
        self.model = model
        self.embedding_model = embedding_model
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.use_images = use_images
        self.pages_per_window = pages_per_window
        self.embedder = LiteLLMEmbedder(model=embedding_model)

    def index(self, content: str, config: IndexingConfig) -> Doc2JSONResult:
        """Sync wrapper for aindex."""
        import asyncio

        return asyncio.run(self.aindex(content, config))

    def _normalize_schema(self, schema: dict) -> dict:
        """Normalize schema to canonical format with 'fields' array.

        Accepts two formats:
        1. Legacy: {"fields": [{"name": "x", "type": "string", ...}]}
        2. Frontend: {"field_name": {"type": "string", "description": "...", ...}}

        Returns canonical format with "fields" array.
        """
        # Already in canonical format
        if "fields" in schema and isinstance(schema["fields"], list):
            return schema

        # Convert frontend format to canonical format
        fields = []
        for name, field_def in schema.items():
            if not isinstance(field_def, dict):
                continue
            field = {
                "name": name,
                "type": field_def.get("type", "string"),
                "description": field_def.get("description", ""),
                "default": field_def.get("default"),
                "examples": field_def.get("examples"),
            }
            # Handle nested objects - "properties" from frontend, "fields" legacy
            nested = field_def.get("properties") or field_def.get("fields")
            if nested and field["type"] == "object":
                normalized_nested = self._normalize_schema(nested)
                field["fields"] = normalized_nested.get("fields", [])
            # Handle array items
            if field["type"] == "array" and "items" in field_def:
                items = field_def["items"]
                if isinstance(items, dict):
                    # Normalize items - could be {"type": "string"} or nested object
                    if "type" in items:
                        field["items"] = items
                        # Recursively normalize nested properties in items
                        nested_props = items.get("properties") or items.get("fields")
                        if nested_props:
                            normalized = self._normalize_schema(nested_props)
                            field["items"]["fields"] = normalized.get("fields", [])
            fields.append(field)

        return {"fields": fields}

    def _build_image_content_blocks(self, images: list[dict]) -> list[dict]:
        """Build multimodal content blocks for LiteLLM from base64-encoded images.

        Args:
            images: List of dicts with keys: content (base64), format (png/jpg), page

        Returns:
            List of image_url content blocks for LLM message.
        """
        blocks = []
        for img in images:
            fmt = img.get("format", "png").lower()
            mime = f"image/{fmt}" if fmt not in ("jpg", "jpeg") else "image/jpeg"
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img['content']}"},
            })
        return blocks

    def _group_images_into_windows(
        self, page_images: list[dict], pages_per_window: int
    ) -> list[list[dict]]:
        """Group page images into windows of N pages each.

        Args:
            page_images: List of image dicts with page numbers, sorted by page.
            pages_per_window: Number of pages per window.

        Returns:
            List of image groups, each containing up to pages_per_window images.
        """
        # Sort by page number
        sorted_images = sorted(page_images, key=lambda x: x.get("page", 0))
        windows = []
        for i in range(0, len(sorted_images), pages_per_window):
            windows.append(sorted_images[i : i + pages_per_window])
        return windows

    async def _process_window_with_images(
        self,
        images: list[dict],
        window_index: int,
        total_windows: int,
        json_schema: dict,
        current_json: dict,
        model: str,
    ) -> tuple[str, dict, int]:
        """Process a window of page images: extract summary and JSON fields.

        Args:
            images: List of base64-encoded page images for this window.
            window_index: Current window number (0-indexed).
            total_windows: Total number of windows.
            json_schema: Normalized JSON schema.
            current_json: Current accumulated extraction state.
            model: LLM model to use (must be multimodal-capable).

        Returns:
            Tuple of (summary, extracted_fields, tokens_used)
        """
        schema_description = self._build_schema_description(json_schema)
        response_schema = self._build_response_schema(json_schema)

        page_nums = [img.get("page", 0) for img in images]
        pages_str = f"pages {min(page_nums)}-{max(page_nums)}" if page_nums else "pages"

        system_prompt = """You are a precise document analyzer that performs two tasks:
1. Generate a concise summary (2-3 sentences) for the current page images
2. Extract and update structured data according to the provided schema

RULES:
- For extraction: only update fields where you find clear evidence in the page images
- For summary: focus on key information relevant to the schema fields
- Return valid JSON that exactly matches the output schema
- Use null for fields with no evidence in these pages
- For arrays: include only NEW items found in these pages (they will be appended)"""

        user_prompt = f"""## Document Context
Window: {window_index + 1} of {total_windows} ({pages_str})

## Schema Definition
{schema_description}

## Current Accumulated State
{json.dumps(current_json, indent=2)}

## Output Format
Return a JSON object with exactly two keys:
{{
  "summary": "A 2-3 sentence summary of these pages' key information",
  "extraction": {{
    // Updated values for schema fields found in these pages
    // Use null if no new information found for a field
    // For arrays, include only NEW items to append
  }}
}}

Analyze the page images below and extract the relevant information."""

        # Build multimodal message with text + images
        image_blocks = self._build_image_content_blocks(images)
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        content.extend(image_blocks)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        # Use structured output if supported
        response_format = None
        if supports_response_schema(model=model):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "doc2json_extraction",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

        # Call LLM with retries
        for attempt in range(1, DOC2JSON_MAX_RETRIES + 1):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=DOC2JSON_EXTRACTION_MAX_TOKENS,
                    response_format=response_format,
                    drop_params=True,
                )

                raw = response.choices[0].message.content.strip()
                tokens_used = response.usage.prompt_tokens if response.usage else 0

                # Parse response
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        if attempt < DOC2JSON_MAX_RETRIES:
                            logger.warning(
                                "JSON parse failed attempt %d/%d",
                                attempt,
                                DOC2JSON_MAX_RETRIES,
                            )
                            continue
                        raise

                summary = parsed.get("summary", "")
                extraction = parsed.get("extraction", {})
                extraction = self._validate_extraction(extraction, json_schema)

                return summary, extraction, tokens_used

            except Exception as e:
                if attempt < DOC2JSON_MAX_RETRIES:
                    logger.warning(
                        "Image window processing failed attempt %d/%d: %s",
                        attempt,
                        DOC2JSON_MAX_RETRIES,
                        e,
                    )
                    continue
                raise

        return "", {}, 0

    async def aindex(
        self,
        content: str,
        config: IndexingConfig,
        source_id: str | None = None,
    ) -> Doc2JSONResult:
        """
        Index content using sliding window JSON extraction.

        Supports two modes:
        1. Text-based (default): Process extracted text with token-based windows
        2. Image-based: Process original page images with page-based windows

        Args:
            content: Document text to process (used in text mode or as fallback).
            config: Indexing configuration with json_schema in extra.
                    For image mode, also requires page_images in extra.
            source_id: Optional source identifier.

        Returns:
            Doc2JSONResult with extracted JSON, summary, and embeddings.
        """
        extra = config.extra or {}
        raw_schema = extra.get("json_schema", {"fields": []})
        json_schema = self._normalize_schema(raw_schema)
        model = extra.get("extraction_model", extra.get("model", self.model))

        # Check for image-based processing mode
        use_images = extra.get("use_images", self.use_images)
        page_images = extra.get("page_images", [])
        pages_per_window = extra.get("pages_per_window", self.pages_per_window)

        # Initialize JSON with defaults from schema
        extracted_json = self._init_json_from_schema(json_schema)
        window_summaries: list[dict] = []
        total_input_tokens = 0

        if use_images and page_images:
            # IMAGE-BASED MODE: Process page images
            logger.info(
                "Doc2JSON using image mode: %d pages, %d pages/window",
                len(page_images),
                pages_per_window,
            )

            image_windows = self._group_images_into_windows(page_images, pages_per_window)

            if not image_windows:
                empty_summary = "Empty document with no pages."
                empty_embedding = await self.embedder.aembed(empty_summary)
                return Doc2JSONResult(
                    extracted_json=extracted_json,
                    combined_summary=empty_summary,
                    combined_summary_embedding=empty_embedding,
                    window_summaries=[],
                    json_schema=json_schema,
                    stats={"window_count": 0, "input_tokens": 0, "use_images": True},
                )

            for i, img_window in enumerate(image_windows):
                try:
                    summary, extraction, tokens_used = await self._process_window_with_images(
                        images=img_window,
                        window_index=i,
                        total_windows=len(image_windows),
                        json_schema=json_schema,
                        current_json=extracted_json,
                        model=model,
                    )

                    extracted_json = self._merge_json(extracted_json, extraction)
                    pages_in_window = [img.get("page", "?") for img in img_window]

                    window_summaries.append({
                        "window_idx": i,
                        "pages": pages_in_window,
                        "summary": summary,
                        "extracted_fields": list(extraction.keys()) if extraction else [],
                        "tokens_used": tokens_used,
                    })
                    total_input_tokens += tokens_used

                    logger.debug(
                        "Image window %d/%d (pages %s): extracted %d fields",
                        i + 1,
                        len(image_windows),
                        pages_in_window,
                        len(extraction) if extraction else 0,
                    )

                except Exception as e:
                    logger.warning("Image window %d failed: %s", i, e)
                    window_summaries.append({
                        "window_idx": i,
                        "summary": "",
                        "error": str(e),
                    })

            stats = {
                "window_count": len(image_windows),
                "input_tokens": total_input_tokens,
                "pages_per_window": pages_per_window,
                "total_pages": len(page_images),
                "use_images": True,
            }

        else:
            # TEXT-BASED MODE: Process extracted text with token windows
            window_size = extra.get("window_size", self.window_size)
            window_overlap = extra.get("window_overlap", self.window_overlap)

            chunker = FixedSizeChunking(chunk_size=window_size, overlap=window_overlap)
            windows = chunker.chunk(content, source_id=source_id)

            if not windows:
                empty_summary = "Empty document with no content."
                empty_embedding = await self.embedder.aembed(empty_summary)
                return Doc2JSONResult(
                    extracted_json=extracted_json,
                    combined_summary=empty_summary,
                    combined_summary_embedding=empty_embedding,
                    window_summaries=[],
                    json_schema=json_schema,
                    stats={"window_count": 0, "input_tokens": 0, "use_images": False},
                )

            for i, window in enumerate(windows):
                try:
                    summary, extraction, tokens_used = await self._process_window(
                        window_text=window.text,
                        window_index=i,
                        total_windows=len(windows),
                        json_schema=json_schema,
                        current_json=extracted_json,
                        model=model,
                    )

                    extracted_json = self._merge_json(extracted_json, extraction)

                    window_summaries.append({
                        "window_idx": i,
                        "summary": summary,
                        "extracted_fields": list(extraction.keys()) if extraction else [],
                        "tokens_used": tokens_used,
                    })
                    total_input_tokens += tokens_used

                    logger.debug(
                        "Window %d/%d: extracted %d fields",
                        i + 1,
                        len(windows),
                        len(extraction) if extraction else 0,
                    )

                except Exception as e:
                    logger.warning("Window %d failed: %s", i, e)
                    window_summaries.append({
                        "window_idx": i,
                        "summary": "",
                        "error": str(e),
                    })

            stats = {
                "window_count": len(windows),
                "input_tokens": total_input_tokens,
                "summary_tokens": 0,  # Will be updated below
                "window_size": window_size,
                "window_overlap": window_overlap,
                "use_images": False,
            }

        # Generate combined summary
        combined_summary = await self._generate_combined_summary(
            window_summaries=window_summaries,
            extracted_json=extracted_json,
            json_schema=json_schema,
            model=model,
        )

        # Embed combined summary
        combined_summary_embedding = await self.embedder.aembed(combined_summary)

        # Add summary token count to stats
        stats["summary_tokens"] = len(combined_summary.split())

        return Doc2JSONResult(
            extracted_json=extracted_json,
            combined_summary=combined_summary,
            combined_summary_embedding=combined_summary_embedding,
            window_summaries=window_summaries,
            json_schema=json_schema,
            stats=stats,
        )

    def _init_json_from_schema(self, schema: dict) -> dict:
        """Initialize JSON object with default values from schema.

        Note: Schema should already be normalized via _normalize_schema().
        """
        result = {}
        fields = schema.get("fields", [])

        for field_def in fields:
            name = field_def.get("name")
            if not name:
                continue

            default = field_def.get("default")
            field_type = field_def.get("type", "string")

            if default is not None:
                result[name] = default
            elif field_type == "array":
                result[name] = []
            elif field_type == "object":
                # Check both "fields" (canonical) and "properties" (frontend) keys
                nested_fields = field_def.get("fields", field_def.get("properties", []))
                result[name] = self._init_json_from_schema({"fields": nested_fields})
            else:
                result[name] = None

        return result

    async def _process_window(
        self,
        window_text: str,
        window_index: int,
        total_windows: int,
        json_schema: dict,
        current_json: dict,
        model: str,
    ) -> tuple[str, dict, int]:
        """
        Process a single window: extract summary and JSON fields.

        Returns:
            Tuple of (summary, extracted_fields, tokens_used)
        """
        schema_description = self._build_schema_description(json_schema)
        response_schema = self._build_response_schema(json_schema)

        system_prompt = """You are a precise document analyzer that performs two tasks:
1. Generate a concise summary (2-3 sentences) for the current text window
2. Extract and update structured data according to the provided schema

RULES:
- For extraction: only update fields where you find clear evidence in the current window
- For summary: focus on key information relevant to the schema fields
- Return valid JSON that exactly matches the output schema
- Use null for fields with no evidence in this window
- For arrays: include only NEW items found in this window (they will be appended)"""

        user_prompt = f"""## Document Context
Window: {window_index + 1} of {total_windows}

## Current Text Window
---
{window_text}
---

## Schema Definition
{schema_description}

## Current Accumulated State
{json.dumps(current_json, indent=2)}

## Output Format
Return a JSON object with exactly two keys:
{{
  "summary": "A 2-3 sentence summary of this window's key information",
  "extraction": {{
    // Updated values for schema fields found in this window
    // Use null if no new information found for a field
    // For arrays, include only NEW items to append
  }}
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use structured output if supported
        response_format = None
        if supports_response_schema(model=model):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "doc2json_extraction",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        else:
            response_format = {"type": "json_object"}

        # Call LLM with retries
        for attempt in range(1, DOC2JSON_MAX_RETRIES + 1):
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=DOC2JSON_EXTRACTION_MAX_TOKENS,
                    response_format=response_format,
                    drop_params=True,
                )

                raw = response.choices[0].message.content.strip()
                tokens_used = response.usage.prompt_tokens if response.usage else 0

                # Parse response
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    # Fallback: try ast.literal_eval
                    try:
                        parsed = ast.literal_eval(raw)
                    except (ValueError, SyntaxError):
                        if attempt < DOC2JSON_MAX_RETRIES:
                            logger.warning(
                                "JSON parse failed attempt %d/%d",
                                attempt,
                                DOC2JSON_MAX_RETRIES,
                            )
                            continue
                        raise

                summary = parsed.get("summary", "")
                extraction = parsed.get("extraction", {})

                # Validate extraction against schema
                extraction = self._validate_extraction(extraction, json_schema)

                return summary, extraction, tokens_used

            except Exception as e:
                if attempt < DOC2JSON_MAX_RETRIES:
                    logger.warning(
                        "Window processing failed attempt %d/%d: %s",
                        attempt,
                        DOC2JSON_MAX_RETRIES,
                        e,
                    )
                    continue
                raise

        # Should not reach here
        return "", {}, 0

    def _build_schema_description(self, schema: dict) -> str:
        """Build human-readable schema description for the prompt."""
        lines = ["Fields to Extract:"]
        fields = schema.get("fields", [])

        for i, field_def in enumerate(fields, 1):
            name = field_def.get("name", "unknown")
            field_type = field_def.get("type", "string")
            description = field_def.get("description", "")
            default = field_def.get("default")
            examples = field_def.get("examples", [])

            line = f'{i}. "{name}" ({field_type}) - {description}'
            if default is not None:
                line += f"\n   Default: {json.dumps(default)}"
            if examples:
                line += f"\n   Examples: {json.dumps(examples)}"

            # Handle nested objects
            if field_type == "object" and "fields" in field_def:
                line += "\n   Nested fields:"
                for nested in field_def.get("fields", []):
                    nested_name = nested.get("name", "")
                    nested_type = nested.get("type", "string")
                    nested_desc = nested.get("description", "")
                    line += f"\n     - {nested_name} ({nested_type}): {nested_desc}"

            # Handle arrays
            if field_type == "array":
                item_type = field_def.get("item_type", "string")
                line += f"\n   Item type: {item_type}"
                if "items" in field_def and isinstance(field_def["items"], dict):
                    # Array of objects
                    item_fields = field_def["items"].get("fields", [])
                    if item_fields:
                        line += "\n   Item fields:"
                        for item_field in item_fields:
                            if_name = item_field.get("name", "")
                            if_type = item_field.get("type", "string")
                            if_desc = item_field.get("description", "")
                            line += f"\n     - {if_name} ({if_type}): {if_desc}"

            lines.append(line)

        return "\n".join(lines)

    def _build_response_schema(self, user_schema: dict) -> dict:
        """Build JSON Schema for structured output response format."""
        extraction_properties = {}
        required_fields = []

        for field_def in user_schema.get("fields", []):
            name = field_def.get("name")
            if not name:
                continue

            field_type = field_def.get("type", "string")
            json_type = self._map_type_to_json_schema(field_type, field_def)
            extraction_properties[name] = json_type
            required_fields.append(name)

        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "extraction": {
                    "type": "object",
                    "properties": extraction_properties,
                    "required": required_fields,
                    "additionalProperties": False,
                },
            },
            "required": ["summary", "extraction"],
            "additionalProperties": False,
        }

    def _map_type_to_json_schema(self, field_type: str, field_def: dict) -> dict:
        """Map user field type to JSON Schema type."""
        if field_type == "string":
            return {"type": ["string", "null"]}
        elif field_type == "number":
            return {"type": ["number", "null"]}
        elif field_type == "integer":
            return {"type": ["integer", "null"]}
        elif field_type == "boolean":
            return {"type": ["boolean", "null"]}
        elif field_type == "array":
            item_type = field_def.get("item_type", "string")
            if item_type == "object" and "items" in field_def:
                # Array of objects
                item_schema = self._build_object_schema(field_def["items"])
                return {"type": "array", "items": item_schema}
            else:
                return {
                    "type": "array",
                    "items": self._map_type_to_json_schema(item_type, {}),
                }
        elif field_type == "object":
            return self._build_object_schema(field_def)
        else:
            return {"type": ["string", "null"]}

    def _build_object_schema(self, field_def: dict) -> dict:
        """Build JSON Schema for nested object."""
        properties = {}
        required = []

        for nested in field_def.get("fields", []):
            name = nested.get("name")
            if name:
                nested_type = nested.get("type", "string")
                properties[name] = self._map_type_to_json_schema(nested_type, nested)
                required.append(name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _validate_extraction(self, extraction: dict, schema: dict) -> dict:
        """Validate and coerce extracted values to match schema types."""
        validated = {}
        fields = {f.get("name"): f for f in schema.get("fields", []) if f.get("name")}

        for name, value in extraction.items():
            if name not in fields:
                continue

            field_def = fields[name]
            validated[name] = self._validate_field_value(value, field_def)

        return validated

    def _validate_field_value(self, value: Any, field_def: dict) -> Any:
        """Validate a single field value against its definition."""
        if value is None:
            return None

        field_type = field_def.get("type", "string")

        if field_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "yes", "1")
            return bool(value)

        elif field_type in ("number", "integer"):
            try:
                return float(value) if field_type == "number" else int(value)
            except (ValueError, TypeError):
                return None

        elif field_type == "array":
            if not isinstance(value, list):
                return []
            # Validate each item if items schema exists
            items_def = field_def.get("items", {})
            if items_def:
                return [
                    self._validate_field_value(item, items_def)
                    for item in value
                    if item is not None
                ]
            return value

        elif field_type == "object":
            if not isinstance(value, dict):
                return {}
            # Recursively validate nested fields
            nested_schema = {"fields": field_def.get("fields", [])}
            return self._validate_extraction(value, nested_schema)

        else:  # string
            return str(value) if value is not None else None

    def _merge_json(self, base: dict, updates: dict) -> dict:
        """
        Merge updates into base using last-wins for scalars, append for arrays.

        Args:
            base: Current accumulated JSON state.
            updates: New extractions from current window.

        Returns:
            Merged JSON object.
        """
        result = dict(base)

        for key, value in updates.items():
            if value is None:
                # Skip nulls (no evidence in this window)
                continue

            if key not in result:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result[key], dict):
                # Deep merge objects
                result[key] = self._merge_json(result[key], value)
            elif isinstance(value, list) and isinstance(result[key], list):
                # Append new array items
                result[key] = result[key] + value
            else:
                # Last value wins for scalars
                result[key] = value

        return result

    async def _generate_combined_summary(
        self,
        window_summaries: list[dict],
        extracted_json: dict,
        json_schema: dict,
        model: str,
    ) -> str:
        """Generate a combined summary from all window summaries."""
        # Collect non-empty summaries
        summaries = [
            ws.get("summary", "")
            for ws in window_summaries
            if ws.get("summary")
        ]

        if not summaries:
            # Fallback: generate summary from extracted JSON
            return f"Document summary: {json.dumps(extracted_json, indent=2)}"

        if len(summaries) == 1:
            return summaries[0]

        # Combine summaries using LLM
        combined_text = "\n".join(f"- {s}" for s in summaries)

        system_prompt = """You are a document summarizer. Given per-section summaries from a document,
create a single coherent summary that captures the key information. Be concise (2-4 sentences)."""

        user_prompt = f"""## Section Summaries
{combined_text}

## Extracted Data
{json.dumps(extracted_json, indent=2)}

Generate a concise summary that captures the most important information from this document."""

        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=DOC2JSON_SUMMARY_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("Combined summary generation failed: %s", e)
            # Fallback: concatenate first few summaries
            return " ".join(summaries[:3])
