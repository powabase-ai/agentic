"""Test image extraction with vision models via LiteLLM

This demonstrates how to use the ImageExtractor to extract text descriptions
from images using various vision-capable models.
"""

import os
from agentic.etl import Pipeline

# Example 1: Extract from image with default settings (gpt-4o)
def test_basic_image_extraction():
    """Basic image extraction with default GPT-4o model"""
    print("=" * 80)
    print("Test 1: Basic Image Extraction (GPT-4o)")
    print("=" * 80)
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/image.jpg")
        .extract(
            format="image",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000,
            temperature=0.7,
        )
    )
    
    result = pipeline.process("path/to/image.jpg")
    
    print(f"Extracted Text: {result['extraction']['text'][:200]}...")
    print(f"Title: {result['extraction']['metadata']['title']}")
    print(f"Short Description: {result['extraction']['metadata']['short_description']}")
    print(f"Tokens Used: {result['extraction']['metadata'].get('tokens', 'N/A')}")
    print()


# Example 2: Auto-detect image format
def test_auto_detect_image():
    """Auto-detect image format and extract"""
    print("=" * 80)
    print("Test 2: Auto-Detect Image Format")
    print("=" * 80)
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/image.png")
        .extract(
            format="auto",  # Will auto-detect as image
            file_path="path/to/image.png",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )
    
    result = pipeline.process("path/to/image.png")
    
    print(f"Extraction Method: {result['extraction']['metadata']['extraction_method']}")
    print(f"Model: {result['extraction']['metadata']['model']}")
    print(f"Extracted Text: {result['extraction']['text'][:200]}...")
    print()


# Example 3: Use Claude 3 for image extraction
def test_claude_vision():
    """Use Anthropic Claude 3 for image extraction via LiteLLM"""
    print("=" * 80)
    print("Test 3: Claude 3 Vision")
    print("=" * 80)
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/image.jpg")
        .extract(
            format="image",
            model="claude-3-sonnet-20240229",  # or claude-3-opus, claude-3-haiku
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=1000,
        )
    )
    
    result = pipeline.process("path/to/image.jpg")
    
    print(f"Model: {result['extraction']['metadata']['model']}")
    print(f"Extracted Text: {result['extraction']['text'][:200]}...")
    print()


# Example 4: Use Google Gemini for image extraction
def test_gemini_vision():
    """Use Google Gemini for image extraction via LiteLLM"""
    print("=" * 80)
    print("Test 4: Google Gemini Vision")
    print("=" * 80)
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/image.jpg")
        .extract(
            format="image",
            model="gemini-1.5-pro",  # or gemini-pro-vision
            api_key=os.getenv("GEMINI_API_KEY"),
            max_tokens=1000,
        )
    )
    
    result = pipeline.process("path/to/image.jpg")
    
    print(f"Model: {result['extraction']['metadata']['model']}")
    print(f"Extracted Text: {result['extraction']['text'][:200]}...")
    print()


# Example 5: Full pipeline with chunking and embedding
def test_full_image_pipeline():
    """Complete ETL pipeline for image: extract -> chunk -> embed"""
    print("=" * 80)
    print("Test 5: Full Image ETL Pipeline")
    print("=" * 80)
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/image.jpg")
        .extract(
            format="auto",
            file_path="path/to/image.jpg",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        .chunk(
            strategy="recursive",
            size=500,
            overlap=100,
        )
        .embed(
            provider="openai",
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    )
    
    result = pipeline.process("path/to/image.jpg")
    
    print(f"Extracted Text Length: {len(result['extraction']['text'])} chars")
    print(f"Number of Chunks: {len(result['chunks'])}")
    print(f"First Chunk: {result['chunks'][0]['text'][:100]}...")
    print(f"Has Embeddings: {result['chunks'][0].get('embedding') is not None}")
    print()


# Example 6: Custom prompt template
def test_custom_prompt():
    """Use a custom prompt template for specific extraction needs"""
    print("=" * 80)
    print("Test 6: Custom Prompt Template")
    print("=" * 80)
    
    custom_prompt = """
    Analyze this medical image with filename "{filename}" and provide:
    1. A title describing the type of medical document or image
    2. A brief one-sentence description
    3. A detailed description including:
       - Type of medical document (X-ray, MRI, prescription, medical certificate, etc.)
       - Visible text and information
       - Any medical conditions, dates, or names mentioned
       - Relevant medical codes or terminology
    
    Respond in JSON format with keys: title, short_description, long_description
    """
    
    pipeline = (
        Pipeline()
        .source("file", path="path/to/medical_image.jpg")
        .extract(
            format="image",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY"),
            prompt_template=custom_prompt,
            max_tokens=1500,
        )
    )
    
    result = pipeline.process("path/to/medical_image.jpg")
    
    print(f"Title: {result['extraction']['metadata']['title']}")
    print(f"Extracted Text: {result['extraction']['text'][:300]}...")
    print()


if __name__ == "__main__":
    print("Image Extractor Test Examples")
    print("=" * 80)
    print()
    print("NOTE: These are example functions. To run them, you need to:")
    print("1. Set up your API keys as environment variables")
    print("2. Provide actual image file paths")
    print("3. Uncomment and call the test functions you want to run")
    print()
    print("Supported models via LiteLLM:")
    print("- OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4-vision-preview")
    print("- Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku")
    print("- Google: gemini-1.5-pro, gemini-pro-vision")
    print("- And many more: https://docs.litellm.ai/docs/providers")
    print()
    
    # Uncomment to run specific tests:
    # test_basic_image_extraction()
    # test_auto_detect_image()
    # test_claude_vision()
    # test_gemini_vision()
    # test_full_image_pipeline()
    # test_custom_prompt()

