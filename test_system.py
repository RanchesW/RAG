#!/usr/bin/env python3
"""Test Kazakhstan RAG System Components"""

def test_gpu():
    import torch
    print("=== GPU Test ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")

def test_embeddings():
    print("\n=== Testing Nomic Embeddings ===")
    try:
        from src.embeddings.nomic_embeddings import NomicEmbeddings
        embeddings = NomicEmbeddings()
        test_text = "Это тест русского текста"
        result = embeddings.encode([test_text])
        print(f"✓ Embeddings working: {result.shape}")
    except Exception as e:
        print(f"✗ Embeddings failed: {e}")

def test_qdrant():
    print("\n=== Testing Qdrant ===")
    try:
        from src.rag.retrieval import QdrantRetriever
        retriever = QdrantRetriever()
        info = retriever.get_collection_info()
        print(f"✓ Qdrant working: {info}")
    except Exception as e:
        print(f"✗ Qdrant failed: {e}")

if __name__ == "__main__":
    test_gpu()
    test_embeddings()
    test_qdrant()
    print("\n🇰🇿 Basic tests completed!")
