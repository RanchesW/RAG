#!/usr/bin/env python3
"""
Kazakhstan RAG System - Main Application
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def start_api():
    """Start FastAPI server"""
    import uvicorn
    from src.api.routes import app
    from config.settings import settings
    
    print("🇰🇿 Starting Kazakhstan RAG API...")
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level="info"
    )

def start_ui():
    """Start Streamlit UI"""
    import subprocess
    from config.settings import settings
    
    print("🇰🇿 Starting Kazakhstan RAG UI...")
    subprocess.run([
        "streamlit", "run", "ui/streamlit_app.py",
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.address", "0.0.0.0"
    ])

def index_documents(directory: str):
    """Index all documents in directory"""
    from src.rag.pipeline import KazakhstanRAGPipeline
    from config.settings import settings
    
    config = {
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "qdrant_path": str(settings.QDRANT_DIR),
        "collection_name": settings.QDRANT_COLLECTION,
        "tensor_parallel_size": settings.TENSOR_PARALLEL_SIZE
    }
    
    pipeline = KazakhstanRAGPipeline(config)
    
    doc_dir = Path(directory)
    if not doc_dir.exists():
        print(f"Directory not found: {directory}")
        return
    
    print(f"Indexing documents from: {directory}")
    
    for file_path in doc_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.pptx', '.xlsx', '.txt']:
            print(f"Processing: {file_path}")
            success = pipeline.index_document(str(file_path))
            if success:
                print(f"✓ Indexed: {file_path.name}")
            else:
                print(f"✗ Failed: {file_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Kazakhstan RAG System")
    parser.add_argument("command", choices=["api", "ui", "index"], 
                       help="Command to run")
    parser.add_argument("--directory", "-d", 
                       help="Directory to index documents from")
    
    args = parser.parse_args()
    
    if args.command == "api":
        start_api()
    elif args.command == "ui":
        start_ui()
    elif args.command == "index":
        if not args.directory:
            print("Please specify directory with --directory")
            sys.exit(1)
        index_documents(args.directory)

if __name__ == "__main__":
    main()
