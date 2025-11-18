# Image Extraction with Vision Models

The ETL pipeline now supports standalone image files using vision-capable AI models via [LiteLLM](https://docs.litellm.ai/docs/completion/usage).

## Supported Image Formats

- `.jpg` / `.jpeg`
- `.png`
- `.gif`
- `.webp`
- `.bmp`
- `.tiff`

## Supported Vision Models

Through LiteLLM, you can use any vision-capable model from these providers:

### OpenAI
- `gpt-4o` (recommended, default)
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4-vision-preview`

### Anthropic
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Google
- `gemini-1.5-pro`
- `gemini-pro-vision`

### Other Providers
See the [LiteLLM providers documentation](https://docs.litellm.ai/docs/providers) for a complete list.

## How It Works

The `ImageExtractor` uses vision models to:

1. **Encode the image** as base64
2. **Send to vision model** with a structured prompt
3. **Extract structured data**:
   - `title`: An appropriate title for the image
   - `short_description`: A 1-sentence description (e.g., "a photo of...", "a screenshot of...")
   - `long_description`: Detailed description including visible text, objects, and context
4. **Return the long_description** as the primary text for downstream processing (chunking, embedding)

## Basic Usage

### Auto-Detection

The pipeline automatically detects image files by extension:

```python
from agentic.etl import Pipeline
import os

pipeline = (
    Pipeline()
    .source("file", path="path/to/image.jpg")
    .extract(
        format="auto",  # Automatically detects it's an image
        file_path="path/to/image.jpg",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    .chunk(strategy="recursive", size=500, overlap=100)
    .embed(provider="openai", model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
)

result = pipeline.process("path/to/image.jpg")

print(result['extraction']['text'])
print(result['extraction']['metadata']['title'])
print(result['extraction']['metadata']['short_description'])
```

### Explicit Image Format

You can explicitly specify the image extractor:

```python
pipeline = (
    Pipeline()
    .source("file", path="image.png")
    .extract(
        format="image",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=1000,
        temperature=0.7,
    )
)

result = pipeline.process("image.png")
```

## Configuration Options

### ImageExtractor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"gpt-4o"` | Vision model name (any LiteLLM-supported model) |
| `api_key` | str | None | API key (or set via environment variables) |
| `api_base` | str | None | Custom API base URL (optional) |
| `max_tokens` | int | `1000` | Maximum tokens in response |
| `temperature` | float | `0.7` | Sampling temperature |
| `prompt_template` | str | See below | Custom prompt template |

### Environment Variables

Set API keys as environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GEMINI_API_KEY="..."

# For vision model configuration in insurance-demo
export VISION_MODEL="gpt-4o"
export VISION_MAX_TOKENS="1000"
export VISION_TEMPERATURE="0.7"
```

## Custom Prompt Templates

You can customize the extraction prompt for specific use cases:

```python
custom_prompt = """
Analyze this medical document with filename "{filename}" and provide:
1. Document type (X-ray, MRI, prescription, certificate, etc.)
2. Visible text and information
3. Any medical conditions, dates, or patient information
4. Medical codes or terminology

Respond in JSON format with keys: title, short_description, long_description
"""

pipeline = (
    Pipeline()
    .source("file", path="medical_image.jpg")
    .extract(
        format="image",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt_template=custom_prompt,
        max_tokens=1500,
    )
)
```

The prompt template must include the `{filename}` placeholder, which will be replaced with the actual filename.

## Using Different Models

### Claude 3

```python
pipeline = (
    Pipeline()
    .source("file", path="image.jpg")
    .extract(
        format="image",
        model="claude-3-sonnet-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=1000,
    )
)
```

### Google Gemini

```python
pipeline = (
    Pipeline()
    .source("file", path="image.jpg")
    .extract(
        format="image",
        model="gemini-1.5-pro",
        api_key=os.getenv("GEMINI_API_KEY"),
        max_tokens=1000,
    )
)
```

## Metadata

The extraction result includes rich metadata:

```python
result = pipeline.process("image.jpg")

metadata = result['extraction']['metadata']
print(metadata['extraction_method'])  # "vision_model"
print(metadata['model'])               # "gpt-4o"
print(metadata['title'])               # Extracted title
print(metadata['short_description'])  # Brief description
print(metadata['long_description'])   # Detailed description
print(metadata['filename'])            # Original filename
print(metadata['mime_type'])           # "image/jpeg"
print(metadata['file_size'])           # File size in bytes
print(metadata['tokens'])              # Total tokens used
print(metadata['prompt_tokens'])       # Prompt tokens
print(metadata['completion_tokens'])   # Completion tokens
```

## Insurance Demo Integration

The insurance-demo backend automatically handles images:

### Upload an Image

```bash
curl -X POST \
  -F "file=@medical_certificate.jpg" \
  http://localhost:5000/documents/upload
```

### Response

```json
{
  "document_id": 123,
  "task_id": "abc123...",
  "status": "pending",
  "file_type": "jpg",
  "is_image": true
}
```

The system will:
1. Detect that it's an image file
2. Use the configured vision model (default: `gpt-4o`)
3. Extract text description via vision API
4. Chunk and embed the description
5. Store in the database with full metadata

### Check Processing Status

```bash
curl http://localhost:5000/documents/123/status
```

## Cost Considerations

Vision API calls are more expensive than text-only models:

- **GPT-4o**: ~$0.005 per image (varies by resolution)
- **GPT-4o-mini**: ~$0.0015 per image (cheaper alternative)
- **Claude 3**: Pricing varies by model tier

Consider:
1. Using `gpt-4o-mini` for lower-priority images
2. Implementing rate limiting for image uploads
3. Caching results for identical images
4. Setting `max_tokens` appropriately to control costs

## Error Handling

The ImageExtractor includes robust error handling:

```python
try:
    result = pipeline.process("image.jpg")
except RuntimeError as e:
    print(f"Image extraction failed: {e}")
```

Common errors:
- **Invalid API key**: Set the correct API key for your chosen model
- **Unsupported format**: Ensure the file is a supported image format
- **Token limit exceeded**: Increase `max_tokens` or use a simpler prompt
- **Model not available**: Check LiteLLM documentation for model availability

## Best Practices

1. **Choose the right model**:
   - Use `gpt-4o` for complex images with text
   - Use `gpt-4o-mini` for simple images
   - Use `claude-3-sonnet` for medical/technical documents

2. **Optimize prompts**:
   - Be specific about what information to extract
   - Use structured JSON output
   - Request only necessary details

3. **Handle failures gracefully**:
   - Implement retry logic with exponential backoff
   - Fall back to alternative models if primary fails
   - Log failures for analysis

4. **Monitor costs**:
   - Track token usage per image
   - Set up alerts for unusual usage patterns
   - Consider implementing daily/monthly quotas

## Testing

See `test_image_extractor.py` for comprehensive examples:

```bash
cd agentic
python test_image_extractor.py
```

## References

- [LiteLLM Documentation](https://docs.litellm.ai/docs/completion/usage)
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Anthropic Claude 3](https://docs.anthropic.com/claude/docs/vision)

