from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import List, Union
import logging

logger = logging.getLogger(__name__)

class NomicEmbeddings:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Nomic embeddings on {self.device}")
        
        try:
            self.model = SentenceTransformer(
                model_name, 
                device=self.device,
                trust_remote_code=True  # Fix for Nomic embeddings
            )
            logger.info(f"✓ Nomic embeddings loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Add prefix for better retrieval (Nomic recommendation)
            prefixed_texts = [f"search_query: {text}" if len(texts) == 1 
                            else f"search_document: {text}" for text in texts]
            
            embeddings = self.model.encode(
                prefixed_texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 10,
                convert_to_numpy=True
            )
            
            logger.debug(f"Encoded {len(texts)} texts to embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
