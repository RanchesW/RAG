import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.embeddings.nomic_embeddings import NomicEmbeddings
from src.llm.deepseek_service import DeepSeekService
from src.documents.processor import DocumentProcessor
from src.rag.retrieval import QdrantRetriever

logger = logging.getLogger(__name__)

class KazakhstanRAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        logger.info("Initializing Kazakhstan RAG Pipeline...")
        
        try:
            # Embeddings
            self.embeddings = NomicEmbeddings(config.get("embedding_model"))
            
            # Vector store
            self.retriever = QdrantRetriever(
                qdrant_path=config.get("qdrant_path"),
                collection_name=config.get("collection_name"),
                vector_size=self.embeddings.get_dimension()
            )
            
            # LLM (only load when needed due to memory requirements)
            self.llm = None
            self.llm_config = {
                "model_name": config.get("llm_model"),
                "tensor_parallel_size": config.get("tensor_parallel_size", 2)
            }
            
            # Document processor
            self.doc_processor = DocumentProcessor()
            
            logger.info("✓ RAG Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def _load_llm(self):
        """Lazy load LLM when needed"""
        if self.llm is None:
            logger.info("Loading DeepSeek LLM...")
            self.llm = DeepSeekService(**self.llm_config)
    
    def index_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """Process and index a document"""
        try:
            logger.info(f"Indexing document: {file_path}")
            
            # Process document
            doc_result = self.doc_processor.process_document(file_path, metadata)
            
            if not doc_result["chunks"]:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            # Create documents for indexing
            documents = []
            for i, chunk in enumerate(doc_result["chunks"]):
                documents.append({
                    "text": chunk,
                    "file_name": doc_result["file_name"],
                    "file_path": doc_result["file_path"],
                    "chunk_index": i,
                    "metadata": doc_result["metadata"]
                })
            
            # Generate embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = self.embeddings.encode(texts)
            
            # Add to vector store
            success = self.retriever.add_documents(documents, embeddings.tolist())
            
            if success:
                logger.info(f"✓ Successfully indexed {len(documents)} chunks from {file_path}")
            else:
                logger.error(f"Failed to index {file_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Document indexing failed: {e}")
            return False
    
    def query(self, 
             question: str, 
             top_k: int = 5,
             filter_conditions: Optional[Dict] = None) -> Dict:
        """Query the RAG system"""
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embeddings.encode([question])[0]
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.search(
                query_embedding=query_embedding.tolist(),
                limit=top_k,
                filter_conditions=filter_conditions
            )
            
            if not retrieved_docs:
                return {
                    "answer": "Извините, я не нашёл релевантной информации в документах.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context
            context_parts = []
            sources = []
            
            for doc in retrieved_docs:
                context_parts.append(doc["text"])
                sources.append({
                    "file_name": doc["file_name"],
                    "score": doc["score"]
                })
            
            context = "\n\n".join(context_parts)
            
            # Load LLM and generate answer
            self._load_llm()
            prompt = self.llm.create_rag_prompt(question, context)
            answer = self.llm.generate(prompt)
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": avg_score,
                "context_used": len(context_parts)
            }
            
            logger.info(f"✓ Generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "answer": "Произошла ошибка при обработке запроса.",
                "sources": [],
                "confidence": 0.0
            }
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            qdrant_info = self.retriever.get_collection_info()
            return {
                "documents_indexed": qdrant_info.get("points_count", 0),
                "embeddings_model": self.embeddings.model_name,
                "llm_loaded": self.llm is not None,
                "collection_status": qdrant_info.get("status", "unknown")
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
