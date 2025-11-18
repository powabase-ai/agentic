#!/usr/bin/env python3
"""Test script for LiteLLM embedders"""

import os
import sys

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agentic.etl.registry import EmbedderFactory


def test_embedder(provider: str, model: str, **kwargs):
    """Test an embedder with sample texts"""
    print(f"\n{'='*60}")
    print(f"Testing: {provider} / {model}")
    print(f"Config: {kwargs}")
    print(f"{'='*60}")
    
    try:
        # Create embedder
        embedder = EmbedderFactory.create(
            provider=provider,
            model=model,
            **kwargs
        )
        print(f"✓ Created embedder: {embedder.__class__.__name__}")
        
        # Test texts
        texts = [
            "LiteLLM provides a unified interface for embeddings.",
            "This makes it easy to switch between providers.",
            "Hello world!"
        ]
        
        # Generate embeddings
        embeddings = embedder.embed(texts)
        
        # Validate results
        assert len(embeddings) == len(texts), "Mismatch in number of embeddings"
        assert all(isinstance(emb, list) for emb in embeddings), "Invalid embedding format"
        assert all(len(emb) > 0 for emb in embeddings), "Empty embeddings"
        assert all(isinstance(val, float) for emb in embeddings for val in emb), "Non-float values in embeddings"
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        print(f"✓ Embedding dimension: {len(embeddings[0])}")
        print(f"✓ Sample embedding (first 5 values): {embeddings[0][:5]}")
        print(f"✅ SUCCESS")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        return False


def main():
    """Run embedder tests"""
    print("LiteLLM Embedder Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: OpenAI (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        results["openai"] = test_embedder(
            provider="openai",
            model="text-embedding-3-small"
        )
    else:
        print("\n⚠️  Skipping OpenAI test (OPENAI_API_KEY not set)")
        results["openai"] = None
    
    # Test 2: OpenAI with custom dimensions
    if os.getenv("OPENAI_API_KEY"):
        results["openai-dimensions"] = test_embedder(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=256
        )
    else:
        results["openai-dimensions"] = None
    
    # Test 3: Cohere (requires COHERE_API_KEY)
    if os.getenv("COHERE_API_KEY"):
        results["cohere"] = test_embedder(
            provider="cohere",
            model="embed-english-v3.0",
            input_type="search_document"
        )
    else:
        print("\n⚠️  Skipping Cohere test (COHERE_API_KEY not set)")
        results["cohere"] = None
    
    # Test 4: Voyage AI (requires VOYAGE_API_KEY)
    if os.getenv("VOYAGE_API_KEY"):
        results["voyage"] = test_embedder(
            provider="voyage",
            model="voyage-01"
        )
    else:
        print("\n⚠️  Skipping Voyage test (VOYAGE_API_KEY not set)")
        results["voyage"] = None
    
    # Test 5: LiteLLM direct usage
    if os.getenv("OPENAI_API_KEY"):
        results["litellm"] = test_embedder(
            provider="litellm",
            model="text-embedding-3-small"
        )
    else:
        results["litellm"] = None
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for name, result in results.items():
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"{status} - {name}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed > 0:
        sys.exit(1)
    
    print("\n✨ All tests passed!")


if __name__ == "__main__":
    main()

