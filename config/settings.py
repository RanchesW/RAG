import os
from pathlib import Path

class Settings:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    QDRANT_DIR = PROJECT_ROOT / "qdrant_db"
    UPLOADS_DIR = PROJECT_ROOT / "uploads"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Model settings
    LLM_MODEL = "deepseek-r1:14b"
    EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    
    # LLM Configuration
    LLM_TEMPERATURE = 0.7
    LLM_TOP_P = 0.9
    LLM_MAX_TOKENS = 2000
    LLM_TIMEOUT = 120
    LLM_MAX_RETRIES = 3
    
    # Ollama settings
    OLLAMA_URL = "http://localhost:11434"
    
    # GPU settings (для embeddings)
    TENSOR_PARALLEL_SIZE = 1
    GPU_MEMORY_UTILIZATION = 0.8
    MAX_MODEL_LEN = 4096
    
    # Vector database
    QDRANT_COLLECTION = "kazakhstan_documents"
    VECTOR_SIZE = 768
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    STREAMLIT_PORT = 8501

settings = Settings()
