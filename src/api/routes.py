from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import sys
import shutil
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.pipeline import KazakhstanRAGPipeline
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Kazakhstan RAG API",
    description="🇰🇿 RAG System for Kazakhstan Corporate Documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag_config = {
    "embedding_model": settings.EMBEDDING_MODEL,
    "llm_model": settings.LLM_MODEL,
    "qdrant_path": str(settings.QDRANT_DIR),
    "collection_name": settings.QDRANT_COLLECTION,
    "tensor_parallel_size": settings.TENSOR_PARALLEL_SIZE
}

pipeline = KazakhstanRAGPipeline(rag_config)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    confidence: float
    context_used: int

@app.get("/")
async def root():
    return {"message": "🇰🇿 Kazakhstan RAG System API", "status": "running"}

@app.get("/health")
async def health_check():
    try:
        stats = pipeline.get_system_stats()
        return {"status": "healthy", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document"""
    try:
        # Check file type
        if not any(file.filename.lower().endswith(ext) 
                  for ext in ['.pdf', '.docx', '.pptx', '.xlsx', '.txt']):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Use: PDF, DOCX, PPTX, XLSX, TXT"
            )
        
        # Save uploaded file
        upload_path = settings.UPLOADS_DIR / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index document
        success = pipeline.index_document(str(upload_path))
        
        if success:
            return {
                "message": f"Document '{file.filename}' uploaded and indexed successfully",
                "file_name": file.filename,
                "status": "indexed"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to index document"
            )
            
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            filter_conditions=request.filters
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        return pipeline.get_system_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
