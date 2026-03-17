"""
Unit tests for Doc2JSONAlgorithm.

Tests cover all pure helper methods (normalize, init, merge, validate,
build_schema_description, build_response_schema, image helpers) and the
async aindex orchestration method with mocked LLM calls.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.knowledge.indexing.doc2json import Doc2JSONAlgorithm, Doc2JSONResult
from agentic.knowledge.models import IndexingConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def algo():
    """Doc2JSONAlgorithm instance with default settings."""
    return Doc2JSONAlgorithm()


@pytest.fixture
def canonical_schema():
    """Schema in canonical {"fields": [...]} format with varied types."""
    return {
        "fields": [
            {"name": "title", "type": "string", "description": "Document title"},
            {"name": "revenue", "type": "number", "description": "Annual revenue"},
            {"name": "page_count", "type": "integer", "description": "Total pages"},
            {"name": "is_public", "type": "boolean", "description": "Public company?"},
            {
                "name": "tags",
                "type": "array",
                "item_type": "string",
                "description": "Tags",
            },
            {
                "name": "address",
                "type": "object",
                "description": "Address",
                "fields": [
                    {"name": "city", "type": "string", "description": "City"},
                    {"name": "zip", "type": "string", "description": "Zip code"},
                ],
            },
            {
                "name": "contacts",
                "type": "array",
                "item_type": "object",
                "description": "Contact list",
                "items": {
                    "type": "object",
                    "fields": [
                        {"name": "name", "type": "string", "description": "Name"},
                        {"name": "email", "type": "string", "description": "Email"},
                    ],
                },
            },
        ]
    }


@pytest.fixture
def frontend_schema():
    """Same data in frontend {"field_name": {"type": ...}} format."""
    return {
        "title": {"type": "string", "description": "Document title"},
        "revenue": {"type": "number", "description": "Annual revenue"},
    }


# ---------------------------------------------------------------------------
# 1. TestNormalizeSchema
# ---------------------------------------------------------------------------

class TestNormalizeSchema:
    """Tests for _normalize_schema."""

    def test_canonical_passthrough(self, algo, canonical_schema):
        """Canonical format passes through unchanged."""
        result = algo._normalize_schema(canonical_schema)
        assert result is canonical_schema

    def test_frontend_single_field(self, algo):
        """Frontend format with one field converts to canonical."""
        schema = {"title": {"type": "string", "description": "Doc title"}}
        result = algo._normalize_schema(schema)
        assert "fields" in result
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "title"
        assert result["fields"][0]["type"] == "string"

    def test_frontend_multiple_fields(self, algo, frontend_schema):
        """Frontend format with multiple fields converts correctly."""
        result = algo._normalize_schema(frontend_schema)
        assert len(result["fields"]) == 2
        names = {f["name"] for f in result["fields"]}
        assert names == {"title", "revenue"}

    def test_nested_object_properties_key(self, algo):
        """Nested objects via 'properties' key normalize recursively."""
        schema = {
            "address": {
                "type": "object",
                "description": "Address",
                "properties": {
                    "city": {"type": "string", "description": "City"},
                },
            }
        }
        result = algo._normalize_schema(schema)
        addr = result["fields"][0]
        assert addr["name"] == "address"
        assert "fields" in addr
        assert addr["fields"][0]["name"] == "city"

    def test_nested_object_fields_key(self, algo):
        """Nested objects via 'fields' key also work."""
        schema = {
            "address": {
                "type": "object",
                "description": "Address",
                "fields": {
                    "city": {"type": "string", "description": "City"},
                },
            }
        }
        result = algo._normalize_schema(schema)
        addr = result["fields"][0]
        assert "fields" in addr
        assert addr["fields"][0]["name"] == "city"

    def test_array_items_with_nested_properties(self, algo):
        """Array items with nested properties normalize."""
        schema = {
            "contacts": {
                "type": "array",
                "description": "Contacts",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name"},
                    },
                },
            }
        }
        result = algo._normalize_schema(schema)
        contacts = result["fields"][0]
        assert contacts["type"] == "array"
        assert "items" in contacts
        assert contacts["items"]["fields"][0]["name"] == "name"

    def test_non_dict_values_skipped(self, algo):
        """Non-dict values in frontend format are silently skipped."""
        schema = {"title": {"type": "string", "description": "x"}, "bad_key": 42}
        result = algo._normalize_schema(schema)
        assert len(result["fields"]) == 1
        assert result["fields"][0]["name"] == "title"


# ---------------------------------------------------------------------------
# 2. TestInitJsonFromSchema
# ---------------------------------------------------------------------------

class TestInitJsonFromSchema:
    """Tests for _init_json_from_schema."""

    def test_scalar_no_default(self, algo):
        """String/number/boolean without default → None."""
        schema = {"fields": [{"name": "x", "type": "string"}]}
        assert algo._init_json_from_schema(schema) == {"x": None}

    def test_explicit_default(self, algo):
        """Explicit defaults are used."""
        schema = {"fields": [{"name": "x", "type": "string", "default": "hello"}]}
        assert algo._init_json_from_schema(schema) == {"x": "hello"}

    def test_array_no_default(self, algo):
        """Array without default → []."""
        schema = {"fields": [{"name": "tags", "type": "array"}]}
        assert algo._init_json_from_schema(schema) == {"tags": []}

    def test_array_with_default(self, algo):
        """Array with explicit default used."""
        schema = {"fields": [{"name": "tags", "type": "array", "default": ["a"]}]}
        assert algo._init_json_from_schema(schema) == {"tags": ["a"]}

    def test_object_recurse(self, algo):
        """Object field recurses into nested defaults."""
        schema = {
            "fields": [
                {
                    "name": "addr",
                    "type": "object",
                    "fields": [{"name": "city", "type": "string"}],
                }
            ]
        }
        result = algo._init_json_from_schema(schema)
        assert result == {"addr": {"city": None}}

    def test_empty_schema(self, algo):
        """Empty schema → {}."""
        assert algo._init_json_from_schema({"fields": []}) == {}


# ---------------------------------------------------------------------------
# 3. TestMergeJson
# ---------------------------------------------------------------------------

class TestMergeJson:
    """Tests for _merge_json."""

    def test_scalar_last_wins(self, algo):
        """Scalar: last-wins overwrites."""
        base = {"name": "old"}
        updates = {"name": "new"}
        assert algo._merge_json(base, updates) == {"name": "new"}

    def test_null_skipped(self, algo):
        """Null values skipped (no overwrite)."""
        base = {"name": "keep"}
        updates = {"name": None}
        assert algo._merge_json(base, updates) == {"name": "keep"}

    def test_arrays_appended(self, algo):
        """Arrays appended."""
        base = {"tags": ["a"]}
        updates = {"tags": ["b", "c"]}
        assert algo._merge_json(base, updates) == {"tags": ["a", "b", "c"]}

    def test_nested_dicts_deep_merged(self, algo):
        """Nested dicts deep-merged."""
        base = {"addr": {"city": "NYC", "zip": None}}
        updates = {"addr": {"zip": "10001"}}
        result = algo._merge_json(base, updates)
        assert result == {"addr": {"city": "NYC", "zip": "10001"}}

    def test_new_keys_added(self, algo):
        """New keys added."""
        base = {"a": 1}
        updates = {"b": 2}
        assert algo._merge_json(base, updates) == {"a": 1, "b": 2}

    def test_empty_updates(self, algo):
        """Empty updates → base unchanged."""
        base = {"a": 1}
        assert algo._merge_json(base, {}) == {"a": 1}


# ---------------------------------------------------------------------------
# 4. TestValidateExtraction
# ---------------------------------------------------------------------------

class TestValidateExtraction:
    """Tests for _validate_extraction."""

    def _schema(self, *fields):
        return {"fields": list(fields)}

    def test_unknown_fields_filtered(self, algo):
        """Unknown fields filtered out."""
        schema = self._schema({"name": "x", "type": "string"})
        result = algo._validate_extraction({"x": "ok", "unknown": "bad"}, schema)
        assert result == {"x": "ok"}
        assert "unknown" not in result

    def test_bool_coerce_true(self, algo):
        """Boolean coercion: 'true'/'yes'/'1' → True (case-insensitive)."""
        schema = self._schema({"name": "flag", "type": "boolean"})
        for val in ("true", "True", "TRUE", "yes", "Yes", "1"):
            result = algo._validate_extraction({"flag": val}, schema)
            assert result["flag"] is True, f"Expected True for {val!r}"

    def test_bool_coerce_false(self, algo):
        """Boolean coercion: 'false'/'no'/'0' → False."""
        schema = self._schema({"name": "flag", "type": "boolean"})
        for val in ("false", "False", "no", "0"):
            result = algo._validate_extraction({"flag": val}, schema)
            assert result["flag"] is False, f"Expected False for {val!r}"

    def test_bool_passthrough(self, algo):
        """Boolean passthrough for actual bools."""
        schema = self._schema({"name": "flag", "type": "boolean"})
        assert algo._validate_extraction({"flag": True}, schema)["flag"] is True
        assert algo._validate_extraction({"flag": False}, schema)["flag"] is False

    def test_number_coercion(self, algo):
        """Number coercion: '42.5' → 42.5."""
        schema = self._schema({"name": "val", "type": "number"})
        result = algo._validate_extraction({"val": "42.5"}, schema)
        assert result["val"] == 42.5

    def test_integer_coercion(self, algo):
        """Integer coercion: '42' → 42."""
        schema = self._schema({"name": "val", "type": "integer"})
        result = algo._validate_extraction({"val": "42"}, schema)
        assert result["val"] == 42

    def test_invalid_number_returns_none(self, algo):
        """Invalid number string → None."""
        schema = self._schema({"name": "val", "type": "number"})
        result = algo._validate_extraction({"val": "not_a_number"}, schema)
        assert result["val"] is None

    def test_string_coercion(self, algo):
        """String coercion: 123 → '123'."""
        schema = self._schema({"name": "x", "type": "string"})
        result = algo._validate_extraction({"x": 123}, schema)
        assert result["x"] == "123"

    def test_non_list_for_array(self, algo):
        """Non-list for array field → []."""
        schema = self._schema({"name": "tags", "type": "array"})
        result = algo._validate_extraction({"tags": "not_a_list"}, schema)
        assert result["tags"] == []

    def test_nested_object_validated(self, algo):
        """Nested object validated recursively."""
        schema = self._schema(
            {
                "name": "addr",
                "type": "object",
                "fields": [{"name": "zip", "type": "integer"}],
            }
        )
        result = algo._validate_extraction({"addr": {"zip": "90210"}}, schema)
        assert result["addr"]["zip"] == 90210


# ---------------------------------------------------------------------------
# 5. TestBuildSchemaDescription
# ---------------------------------------------------------------------------

class TestBuildSchemaDescription:
    """Tests for _build_schema_description."""

    def test_basic_fields(self, algo):
        """Basic fields listed with numbering, type, description."""
        schema = {
            "fields": [
                {"name": "title", "type": "string", "description": "The title"},
            ]
        }
        desc = algo._build_schema_description(schema)
        assert '"title" (string) - The title' in desc
        assert desc.startswith("Fields to Extract:")

    def test_defaults_and_examples(self, algo):
        """Defaults and examples included when present."""
        schema = {
            "fields": [
                {
                    "name": "status",
                    "type": "string",
                    "description": "Status",
                    "default": "active",
                    "examples": ["active", "inactive"],
                }
            ]
        }
        desc = algo._build_schema_description(schema)
        assert "Default:" in desc
        assert '"active"' in desc
        assert "Examples:" in desc

    def test_nested_object_subfields(self, algo):
        """Nested object sub-fields shown."""
        schema = {
            "fields": [
                {
                    "name": "addr",
                    "type": "object",
                    "description": "Address",
                    "fields": [
                        {"name": "city", "type": "string", "description": "City"},
                    ],
                }
            ]
        }
        desc = algo._build_schema_description(schema)
        assert "Nested fields:" in desc
        assert "city (string): City" in desc

    def test_array_item_type_and_fields(self, algo):
        """Array item type and item fields shown."""
        schema = {
            "fields": [
                {
                    "name": "contacts",
                    "type": "array",
                    "item_type": "object",
                    "description": "Contacts",
                    "items": {
                        "type": "object",
                        "fields": [
                            {"name": "email", "type": "string", "description": "Email"},
                        ],
                    },
                }
            ]
        }
        desc = algo._build_schema_description(schema)
        assert "Item type: object" in desc
        assert "Item fields:" in desc
        assert "email (string): Email" in desc


# ---------------------------------------------------------------------------
# 6. TestBuildResponseSchema
# ---------------------------------------------------------------------------

class TestBuildResponseSchema:
    """Tests for _build_response_schema."""

    def test_scalar_types_nullable(self, algo):
        """String/number/integer/boolean map to nullable JSON Schema types."""
        schema = {
            "fields": [
                {"name": "s", "type": "string"},
                {"name": "n", "type": "number"},
                {"name": "i", "type": "integer"},
                {"name": "b", "type": "boolean"},
            ]
        }
        result = algo._build_response_schema(schema)
        props = result["properties"]["extraction"]["properties"]
        assert props["s"] == {"type": ["string", "null"]}
        assert props["n"] == {"type": ["number", "null"]}
        assert props["i"] == {"type": ["integer", "null"]}
        assert props["b"] == {"type": ["boolean", "null"]}

    def test_array_of_objects(self, algo):
        """Array of objects produces correct nested schema."""
        schema = {
            "fields": [
                {
                    "name": "items",
                    "type": "array",
                    "item_type": "object",
                    "items": {
                        "type": "object",
                        "fields": [
                            {"name": "name", "type": "string"},
                        ],
                    },
                }
            ]
        }
        result = algo._build_response_schema(schema)
        arr = result["properties"]["extraction"]["properties"]["items"]
        assert arr["type"] == "array"
        assert "items" in arr
        assert arr["items"]["type"] == "object"
        assert "name" in arr["items"]["properties"]

    def test_nested_object(self, algo):
        """Nested objects produce correct sub-properties."""
        schema = {
            "fields": [
                {
                    "name": "addr",
                    "type": "object",
                    "fields": [
                        {"name": "city", "type": "string"},
                    ],
                }
            ]
        }
        result = algo._build_response_schema(schema)
        obj = result["properties"]["extraction"]["properties"]["addr"]
        assert obj["type"] == "object"
        assert "city" in obj["properties"]
        assert obj["additionalProperties"] is False

    def test_top_level_structure(self, algo):
        """Top-level has 'summary' + 'extraction' required, additionalProperties: false."""
        schema = {"fields": [{"name": "x", "type": "string"}]}
        result = algo._build_response_schema(schema)
        assert result["required"] == ["summary", "extraction"]
        assert result["additionalProperties"] is False
        assert "summary" in result["properties"]
        assert "extraction" in result["properties"]


# ---------------------------------------------------------------------------
# 7. TestImageHelpers
# ---------------------------------------------------------------------------

class TestImageHelpers:
    """Tests for _build_image_content_blocks and _group_images_into_windows."""

    def test_png_mime_and_data_url(self, algo):
        """PNG → image/png MIME, correct data URL."""
        images = [{"content": "abc123", "format": "png", "page": 1}]
        blocks = algo._build_image_content_blocks(images)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image_url"
        url = blocks[0]["image_url"]["url"]
        assert url == "data:image/png;base64,abc123"

    def test_jpg_jpeg_mime(self, algo):
        """'jpg'/'jpeg' → image/jpeg."""
        for fmt in ("jpg", "jpeg"):
            images = [{"content": "data", "format": fmt, "page": 1}]
            blocks = algo._build_image_content_blocks(images)
            assert "image/jpeg" in blocks[0]["image_url"]["url"]

    def test_even_page_split(self, algo):
        """Even page split into correct window count."""
        pages = [{"page": i, "content": "x", "format": "png"} for i in range(6)]
        windows = algo._group_images_into_windows(pages, pages_per_window=3)
        assert len(windows) == 2
        assert len(windows[0]) == 3
        assert len(windows[1]) == 3

    def test_remainder_handling(self, algo):
        """Remainder handled (7 pages / 3 per window → [3, 3, 1])."""
        pages = [{"page": i, "content": "x", "format": "png"} for i in range(7)]
        windows = algo._group_images_into_windows(pages, pages_per_window=3)
        assert len(windows) == 3
        assert [len(w) for w in windows] == [3, 3, 1]

    def test_out_of_order_sorted(self, algo):
        """Out-of-order pages sorted before grouping."""
        pages = [
            {"page": 5, "content": "x", "format": "png"},
            {"page": 1, "content": "x", "format": "png"},
            {"page": 3, "content": "x", "format": "png"},
        ]
        windows = algo._group_images_into_windows(pages, pages_per_window=2)
        assert len(windows) == 2
        # First window should have pages 1, 3
        assert windows[0][0]["page"] == 1
        assert windows[0][1]["page"] == 3


# ---------------------------------------------------------------------------
# 8. TestAindex (async integration tests with mocked LLM)
# ---------------------------------------------------------------------------

def _make_llm_response(content_dict, prompt_tokens=100):
    """Build a mock LLM response object."""
    mock_resp = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = json.dumps(content_dict)
    mock_resp.choices = [MagicMock(message=mock_msg)]
    mock_resp.usage = MagicMock(prompt_tokens=prompt_tokens)
    return mock_resp


class TestAindex:
    """Async tests for the aindex orchestration method."""

    @pytest.mark.asyncio
    async def test_text_mode_full_flow(self, algo):
        """Text mode: mock LLM → verify Doc2JSONResult fields, merged JSON, stats."""
        schema = {
            "fields": [
                {"name": "title", "type": "string", "description": "Title"},
                {"name": "tags", "type": "array", "item_type": "string", "description": "Tags"},
            ]
        }
        config = IndexingConfig(extra={"json_schema": schema})

        # Window extraction response
        window_resp = _make_llm_response({
            "summary": "This doc is about testing.",
            "extraction": {"title": "Test Doc", "tags": ["unit"]},
        })

        # Combined summary response (returned when >1 summaries, but
        # with short content we may get 1 window — the combined summary
        # path just returns the single summary directly)
        embed_result = [0.1, 0.2, 0.3]

        with (
            patch("agentic.knowledge.indexing.doc2json.litellm") as mock_litellm,
            patch("agentic.knowledge.indexing.doc2json.LiteLLMEmbedder") as MockEmbedder,
            patch("agentic.knowledge.indexing.doc2json.supports_response_schema", return_value=False),
        ):
            mock_litellm.acompletion = AsyncMock(return_value=window_resp)
            mock_embedder_inst = MagicMock()
            mock_embedder_inst.aembed = AsyncMock(return_value=embed_result)
            MockEmbedder.return_value = mock_embedder_inst

            # Re-create algo so embedder mock takes effect
            algo2 = Doc2JSONAlgorithm()

            result = await algo2.aindex(
                content="Some document text that is long enough to create a window.",
                config=config,
            )

        assert isinstance(result, Doc2JSONResult)
        assert result.extracted_json["title"] == "Test Doc"
        assert result.extracted_json["tags"] == ["unit"]
        assert result.combined_summary_embedding == embed_result
        assert result.stats["use_images"] is False
        assert result.stats["window_count"] >= 1

    @pytest.mark.asyncio
    async def test_image_mode_full_flow(self, algo):
        """Image mode: use_images=True + fake page images → verify stats show image mode."""
        schema = {"fields": [{"name": "title", "type": "string", "description": "Title"}]}
        page_images = [
            {"page": i, "content": "base64data", "format": "png"}
            for i in range(3)
        ]
        config = IndexingConfig(
            extra={
                "json_schema": schema,
                "use_images": True,
                "page_images": page_images,
                "pages_per_window": 2,
            }
        )

        window_resp = _make_llm_response({
            "summary": "Image window summary.",
            "extraction": {"title": "From Image"},
        })
        # Combined summary for >1 windows
        combined_resp = MagicMock()
        combined_msg = MagicMock()
        combined_msg.content = "Combined summary of the document."
        combined_resp.choices = [MagicMock(message=combined_msg)]
        combined_resp.usage = MagicMock(prompt_tokens=50)

        embed_result = [0.4, 0.5]

        with (
            patch("agentic.knowledge.indexing.doc2json.litellm") as mock_litellm,
            patch("agentic.knowledge.indexing.doc2json.LiteLLMEmbedder") as MockEmbedder,
            patch("agentic.knowledge.indexing.doc2json.supports_response_schema", return_value=False),
        ):
            # First two calls are window processing, third is combined summary
            mock_litellm.acompletion = AsyncMock(
                side_effect=[window_resp, window_resp, combined_resp]
            )
            mock_embedder_inst = MagicMock()
            mock_embedder_inst.aembed = AsyncMock(return_value=embed_result)
            MockEmbedder.return_value = mock_embedder_inst

            algo2 = Doc2JSONAlgorithm()
            result = await algo2.aindex(content="", config=config)

        assert result.stats["use_images"] is True
        assert result.stats["window_count"] == 2  # 3 pages / 2 per window = 2 windows
        assert result.stats["total_pages"] == 3

    @pytest.mark.asyncio
    async def test_empty_text_content(self):
        """Empty text content returns appropriate message and 0 windows."""
        schema = {"fields": [{"name": "x", "type": "string"}]}
        config = IndexingConfig(extra={"json_schema": schema})

        embed_result = [0.0]

        with (
            patch("agentic.knowledge.indexing.doc2json.litellm"),
            patch("agentic.knowledge.indexing.doc2json.LiteLLMEmbedder") as MockEmbedder,
            patch("agentic.knowledge.indexing.doc2json.supports_response_schema", return_value=False),
        ):
            mock_embedder_inst = MagicMock()
            mock_embedder_inst.aembed = AsyncMock(return_value=embed_result)
            MockEmbedder.return_value = mock_embedder_inst

            algo = Doc2JSONAlgorithm()
            result = await algo.aindex(content="", config=config)

        assert result.combined_summary == "Empty document with no content."
        assert result.stats["window_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_images(self):
        """use_images=True but page_images=[] → falls through to text branch.

        When page_images is an empty list, `use_images and page_images` is
        falsy, so the code takes the text-based path. With empty content,
        we get "Empty document with no content." and use_images=False.
        """
        schema = {"fields": [{"name": "x", "type": "string"}]}
        config = IndexingConfig(
            extra={
                "json_schema": schema,
                "use_images": True,
                "page_images": [],
            }
        )

        embed_result = [0.0]

        with (
            patch("agentic.knowledge.indexing.doc2json.litellm"),
            patch("agentic.knowledge.indexing.doc2json.LiteLLMEmbedder") as MockEmbedder,
            patch("agentic.knowledge.indexing.doc2json.supports_response_schema", return_value=False),
        ):
            mock_embedder_inst = MagicMock()
            mock_embedder_inst.aembed = AsyncMock(return_value=embed_result)
            MockEmbedder.return_value = mock_embedder_inst

            algo = Doc2JSONAlgorithm()
            result = await algo.aindex(content="", config=config)

        # Empty page_images list is falsy → text branch → empty content path
        assert result.combined_summary == "Empty document with no content."
        assert result.stats["window_count"] == 0
        assert result.stats["use_images"] is False

    @pytest.mark.asyncio
    async def test_window_error_handled(self):
        """LLM raises on one window → that window gets 'error' key, result still returned."""
        schema = {"fields": [{"name": "title", "type": "string", "description": "Title"}]}
        page_images = [
            {"page": i, "content": "base64data", "format": "png"}
            for i in range(4)
        ]
        config = IndexingConfig(
            extra={
                "json_schema": schema,
                "use_images": True,
                "page_images": page_images,
                "pages_per_window": 2,
            }
        )

        good_resp = _make_llm_response({
            "summary": "Good window.",
            "extraction": {"title": "Extracted"},
        })

        # Combined summary (single summary → returned directly, but let's
        # provide it in case the code calls LLM for combining)
        combined_resp = MagicMock()
        combined_msg = MagicMock()
        combined_msg.content = "Final summary."
        combined_resp.choices = [MagicMock(message=combined_msg)]
        combined_resp.usage = MagicMock(prompt_tokens=50)

        embed_result = [0.1]

        with (
            patch("agentic.knowledge.indexing.doc2json.litellm") as mock_litellm,
            patch("agentic.knowledge.indexing.doc2json.LiteLLMEmbedder") as MockEmbedder,
            patch("agentic.knowledge.indexing.doc2json.supports_response_schema", return_value=False),
        ):
            # First window succeeds, second fails all retries, then combined summary
            mock_litellm.acompletion = AsyncMock(
                side_effect=[
                    good_resp,  # window 0
                    Exception("LLM Error"),  # window 1, attempt 1
                    Exception("LLM Error"),  # window 1, attempt 2
                    Exception("LLM Error"),  # window 1, attempt 3
                    combined_resp,  # combined summary
                ]
            )
            mock_embedder_inst = MagicMock()
            mock_embedder_inst.aembed = AsyncMock(return_value=embed_result)
            MockEmbedder.return_value = mock_embedder_inst

            algo = Doc2JSONAlgorithm()
            result = await algo.aindex(content="", config=config)

        # Should still get a result
        assert isinstance(result, Doc2JSONResult)
        # One window should have error
        error_windows = [ws for ws in result.window_summaries if "error" in ws]
        assert len(error_windows) >= 1
        # The good window's extraction should still be present
        assert result.extracted_json["title"] == "Extracted"
