from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
import uuid
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class QdrantRetriever:
    def __init__(self, 
                 qdrant_path: str = "./qdrant_db",
                 collection_name: str = "kazakhstan_documents",
                 vector_size: int = 768):
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(path=qdrant_path)
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(collection_name)
                logger.info(f"✓ Connected to existing collection: {collection_name}")
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✓ Created new collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
    
    def get_collection_info(self) -> Dict:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": "green"
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {"error": str(e)}
    
    def add_documents(self, 
                     documents: List[Dict],
                     embeddings: List[List[float]]) -> bool:
        """Add documents with embeddings to Qdrant"""
        try:
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point_id = str(uuid.uuid4())
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc.get("text", ""),
                        "file_name": doc.get("file_name", ""),
                        "file_path": doc.get("file_path", ""),
                        "chunk_index": doc.get("chunk_index", i),
                        "metadata": doc.get("metadata", {})
                    }
                ))
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"✓ Added {len(points)} documents to Qdrant")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(self, 
              query_embedding: List[float],
              limit: int = 5,
              score_threshold: float = 0.3,
              filter_conditions: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        try:
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "file_name": result.payload.get("file_name", ""),
                    "metadata": result.payload.get("metadata", {})
                })
            
            logger.debug(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document(self, file_name: str) -> bool:
        """Delete all chunks of a specific document"""
        try:
            print(f"🔍 Looking for document: {file_name}")
            
            # Используем правильный синтаксис фильтра для Qdrant
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_name",
                            match=MatchValue(value=file_name)
                        )
                    ]
                ),
                limit=1000,
                with_payload=True
            )
            
            points_to_delete = scroll_result[0]
            print(f"📋 Found {len(points_to_delete)} points to delete")
            
            if points_to_delete:
                point_ids = [point.id for point in points_to_delete]
                print(f"🗑️ Deleting point IDs: {point_ids[:5]}...")  # Show first 5
                
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                
                print(f"✅ Successfully deleted {len(point_ids)} chunks")
                logger.info(f"✓ Deleted {len(point_ids)} chunks for document: {file_name}")
                return True
            else:
                print(f"⚠️ No points found for document: {file_name}")
                logger.warning(f"Document not found: {file_name}")
                return False
                
        except Exception as e:
            print(f"❌ Delete failed: {e}")
            logger.error(f"Failed to delete document {file_name}: {e}")
            import traceback
            print(f"🔍 Traceback: {traceback.format_exc()}")
            return False
    
    def delete_all_documents(self) -> bool:
        """Delete all documents from collection"""
        try:
            print("🗑️ Deleting collection and recreating...")
            # Удаляем коллекцию и создаем заново
            self.client.delete_collection(self.collection_name)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            
            print("✅ Collection recreated successfully")
            logger.info("✓ Deleted all documents")
            return True
            
        except Exception as e:
            print(f"❌ Delete all failed: {e}")
            logger.error(f"Failed to delete all documents: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict]:
        """Get list of all unique documents in collection"""
        try:
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            documents = {}
            for point in scroll_result[0]:
                file_name = point.payload.get("file_name", "unknown")
                if file_name not in documents:
                    documents[file_name] = {
                        "file_name": file_name,
                        "chunks": 0
                    }
                documents[file_name]["chunks"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to get documents list: {e}")
            return []
