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

# Настроим логирование чтобы видеть все
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KazakhstanRAGPipeline:
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize components
        print("🔧 Initializing Kazakhstan RAG Pipeline...")
        logger.info("Initializing Kazakhstan RAG Pipeline...")
        
        try:
            # Embeddings
            print("📊 Loading embeddings...")
            self.embeddings = NomicEmbeddings(config.get("embedding_model"))
            
            # Vector store
            print("🗃️ Connecting to Qdrant...")
            self.retriever = QdrantRetriever(
                qdrant_path=config.get("qdrant_path"),
                collection_name=config.get("collection_name"),
                vector_size=self.embeddings.get_dimension()
            )
            
            # LLM
            print("🤖 Loading Ollama LLM...")
            self.llm = DeepSeekService(
                model_name=config.get("llm_model", "deepseek-r1:14b")
            )
            
            # Document processor
            print("📝 Loading document processor...")
            self.doc_processor = DocumentProcessor()
            
            print("✅ RAG Pipeline initialized successfully!")
            logger.info("✓ RAG Pipeline initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            raise
    
    def index_document(self, file_path: str, metadata: Optional[Dict] = None) -> bool:
        """Process and index a document"""
        try:
            print(f"📚 Indexing document: {file_path}")
            logger.info(f"Indexing document: {file_path}")
            
            # Process document
            doc_result = self.doc_processor.process_document(file_path, metadata)
            
            if not doc_result["chunks"]:
                print(f"⚠️ No content extracted from {file_path}")
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
            
            print(f"🔍 Generating embeddings for {len(documents)} chunks...")
            # Generate embeddings
            texts = [doc["text"] for doc in documents]
            embeddings = self.embeddings.encode(texts)
            
            print(f"💾 Adding {len(documents)} documents to Qdrant...")
            # Add to vector store
            success = self.retriever.add_documents(documents, embeddings.tolist())
            
            if success:
                print(f"✅ Successfully indexed {len(documents)} chunks from {file_path}")
                logger.info(f"✓ Successfully indexed {len(documents)} chunks from {file_path}")
            else:
                print(f"❌ Failed to index {file_path}")
                logger.error(f"Failed to index {file_path}")
            
            return success
            
        except Exception as e:
            print(f"❌ Document indexing failed: {e}")
            logger.error(f"Document indexing failed: {e}")
            return False
    
    def query(self, 
             question: str, 
             top_k: int = 5,
             filter_conditions: Optional[Dict] = None) -> Dict:
        """Query the RAG system"""
        try:
            print(f"\n🔍 NEW QUERY: {question}")
            print(f"📊 Searching for top {top_k} results...")
            logger.info(f"Processing query: {question[:100]}...")
            
            # Generate query embedding
            print("🧠 Generating query embedding...")
            query_embedding = self.embeddings.encode([question])[0]
            print(f"✅ Query embedding shape: {query_embedding.shape}")
            
            # Retrieve relevant documents
            print("🔍 Searching in Qdrant...")
            retrieved_docs = self.retriever.search(
                query_embedding=query_embedding.tolist(),
                limit=top_k,
                score_threshold=0.3,
                filter_conditions=filter_conditions
            )
            
            print(f"📋 Found {len(retrieved_docs)} relevant documents")
            for i, doc in enumerate(retrieved_docs):
                print(f"  {i+1}. {doc['file_name']} (score: {doc['score']:.3f})")
                print(f"     Text preview: {doc['text'][:100]}...")
            
            if not retrieved_docs:
                print("❌ No relevant documents found!")
                return {
                    "answer": "Извините, я не нашёл релевантной информации в документах.",
                    "sources": [],
                    "confidence": 0.0,
                    "context_used": 0
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
            print(f"📝 Context length: {len(context)} characters")
            
            # Generate answer using LLM
            print("🤖 Generating answer with Ollama...")
            prompt = self.llm.create_rag_prompt(question, context)
            print(f"📤 Prompt preview: {prompt[:200]}...")
            
            answer = self.llm.generate(prompt)
            print(f"📥 Generated answer: {answer[:200]}...")
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(doc["score"] for doc in retrieved_docs) / len(retrieved_docs)
            print(f"📊 Average confidence: {avg_score:.3f}")
            
            result = {
                "answer": answer,
                "sources": sources,
                "confidence": avg_score,
                "context_used": len(context_parts)
            }
            
            print(f"✅ Query completed successfully!")
            logger.info(f"✓ Generated answer with {len(sources)} sources")
            return result
            
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            logger.error(f"Query processing failed: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return {
                "answer": f"Произошла ошибка при обработке запроса: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "context_used": 0
            }
    
    def delete_document(self, file_name: str) -> bool:
        """Delete a specific document"""
        try:
            print(f"🗑️ Deleting document: {file_name}")
            success = self.retriever.delete_document(file_name)
            
            if success:
                print(f"✅ Document '{file_name}' deleted successfully")
                # Также удаляем файл из uploads если есть
                from config.settings import settings
                file_path = settings.UPLOADS_DIR / file_name
                if file_path.exists():
                    file_path.unlink()
                    print(f"🗑️ Also deleted file: {file_path}")
            else:
                print(f"❌ Failed to delete document: {file_name}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error deleting document: {e}")
            return False
    
    def delete_all_documents(self) -> bool:
        """Delete all documents"""
        try:
            print("🗑️ Deleting ALL documents...")
            success = self.retriever.delete_all_documents()
            
            if success:
                print("✅ All documents deleted successfully")
                # Очищаем папку uploads
                from config.settings import settings
                import shutil
                if settings.UPLOADS_DIR.exists():
                    for file_path in settings.UPLOADS_DIR.iterdir():
                        if file_path.is_file():
                            file_path.unlink()
                    print("🗑️ Also cleared uploads folder")
            else:
                print("❌ Failed to delete all documents")
            
            return success
            
        except Exception as e:
            print(f"❌ Error deleting all documents: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all documents"""
        try:
            return self.retriever.get_all_documents()
        except Exception as e:
            print(f"❌ Error getting documents list: {e}")
            return []
    
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            qdrant_info = self.retriever.get_collection_info()
            stats = {
                "documents_indexed": qdrant_info.get("points_count", 0),
                "embeddings_model": self.embeddings.model_name,
                "llm_loaded": True,
                "llm_model": self.llm.model_name,
                "collection_status": qdrant_info.get("status", "unknown")
            }
            print(f"📊 System stats: {stats}")
            return stats
        except Exception as e:
            print(f"❌ Failed to get system stats: {e}")
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}
