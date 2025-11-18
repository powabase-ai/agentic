# Agentic ETL Package

A flexible, pluggable ETL (Extract, Transform, Load) pipeline for document processing with support for multiple document types including PDFs, DOCX, text files, and **images**.

## Features

- 🔌 **Pluggable Architecture**: Registry-based system for extractors, chunkers, embedders, and sinks
- 📄 **Multiple Document Types**: PDF, DOCX, TXT, MD, and more
- 🖼️ **Image Support**: Extract text from images using vision AI models (GPT-4o, Claude 3, Gemini, etc.)
- 🔄 **Multiple Extraction Strategies**: Fallback mechanisms for robust extraction
- ✂️ **Intelligent Chunking**: Recursive, semantic, and custom chunking strategies
- 🧠 **Embeddings**: Support for OpenAI, Anthropic, and other embedding providers via LiteLLM
- 💾 **Multiple Storage Backends**: PostgreSQL with pgvector, and more
- 🚀 **Async Support**: Built-in async processing capabilities
- ⚙️ **Fluent Builder Pattern**: Easy-to-use pipeline configuration

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### Basic Document Processing

```python
from agentic.etl import Pipeline
import os

# Process a PDF
pipeline = (
    Pipeline()
    .source("file", path="document.pdf")
    .extract(format="auto", file_path="document.pdf")
    .chunk(strategy="recursive", size=1000, overlap=200)
    .embed(
        provider="openai",
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

result = pipeline.process("document.pdf")
print(f"Extracted {len(result['chunks'])} chunks")
```

### Image Processing

```python
# Process an image using vision AI
pipeline = (
    Pipeline()
    .source("file", path="medical_certificate.jpg")
    .extract(
        format="auto",  # Auto-detects image format
        file_path="medical_certificate.jpg",
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    .chunk(strategy="recursive", size=500, overlap=100)
    .embed(provider="openai", model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
)

result = pipeline.process("medical_certificate.jpg")
print(f"Extracted text: {result['extraction']['text']}")
print(f"Title: {result['extraction']['metadata']['title']}")
```

## Supported Formats

### Documents
- **PDF**: Multi-strategy extraction (Mistral OCR → PyMuPDF → pdfplumber)
- **DOCX**: Microsoft Word documents
- **TXT, MD, TEX, RST**: Plain text and markdown files

### Images (NEW!)
- **JPG/JPEG**: JPEG images
- **PNG**: PNG images
- **GIF**: GIF images
- **WEBP**: WebP images
- **BMP**: Bitmap images
- **TIFF**: TIFF images

Images are processed using vision-capable AI models via [LiteLLM](https://docs.litellm.ai/docs/completion/usage).

## Documentation

- [**Image Extraction Guide**](IMAGE_EXTRACTION.md) - Comprehensive guide for image processing with vision models
- [**Test Examples**](test_image_extractor.py) - Example code for various use cases

## Architecture

### Extractors

Convert documents/images to text:

```python
# Auto-detection (recommended)
.extract(format="auto", file_path="document.pdf")

# Explicit format
.extract(format="pdf", mistral_api_key="...")
.extract(format="docx")
.extract(format="image", model="gpt-4o", api_key="...")
```

Available extractors:
- `pdf`: PDFExtractor with fallback strategy
- `docx`: DocxExtractor
- `text`: TextExtractor
- `image`: ImageExtractor (uses vision models)
- `auto`: Auto-detect based on file extension

### Chunkers

Split text into manageable chunks:

```python
.chunk(strategy="recursive", size=1000, overlap=200)
```

Available strategies:
- `recursive`: RecursiveCharacterTextSplitter (langchain)
- `semantic`: Semantic chunking based on meaning
- Custom chunkers can be registered

### Embedders

Generate vector embeddings:

```python
.embed(provider="openai", model="text-embedding-3-small", api_key="...")
```

Supported providers (via LiteLLM):
- OpenAI
- Anthropic
- Cohere
- And many more

### Sinks

Store processed data:

```python
.load("postgresql", connection_string="...")
```

## Advanced Usage

### Custom Extractors

Register your own extractors:

```python
from agentic.etl.registry import ExtractorRegistry
from agentic.etl.transformers.extractors.base import BaseExtractor, ExtractionResult

@ExtractorRegistry.register("custom")
class CustomExtractor(BaseExtractor):
    def extract(self, file_path: str) -> ExtractionResult:
        # Your extraction logic
        text = "..."
        return ExtractionResult(text=text, metadata={})
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export MISTRAL_API_KEY="..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# Image Processing
export VISION_MODEL="gpt-4o"
export VISION_MAX_TOKENS="1000"
export VISION_TEMPERATURE="0.7"
```

## Testing

```bash
# Test embedders
python test_embedders.py

# Test image extraction
python test_image_extractor.py
```

## Integration

This package is designed to be imported by other applications. See the `insurance-demo` project for a complete example of integrating the ETL pipeline with Flask and Celery.

## Dependencies

Key dependencies:
- `litellm`: Universal LLM API client
- `PyMuPDF`: PDF processing
- `pdfplumber`: Alternative PDF processor
- `mistralai`: Mistral OCR for PDFs
- `langchain-text-splitters`: Text chunking
- `tiktoken`: Token counting
- `pgvector`: Vector database support

See `pyproject.toml` for the complete list.

## Contributing

Contributions are welcome! To add new extractors, chunkers, or embedders:

1. Create your component in the appropriate directory
2. Register it with the appropriate registry
3. Add tests
4. Update documentation

## License

[Add your license here]

