# --- The "Custom Knowledge" tool (for Phase 3).
"""
RAG (Retrieval-Augmented Generation) Tool
==========================================
For custom knowledge base integration (Phase 3).
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from loguru import logger


class DocumentStore:
    """
    Document storage for RAG.
    
    Future implementation will include:
    - Vector embeddings
    - Semantic search
    - Document chunking
    - Metadata filtering
    """
    
    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.embeddings_available = False
        
        logger.info("Document store initialized (basic mode)")
    
    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Add a document to the store"""
        
        self.documents[doc_id] = {
            "content": content,
            "metadata": metadata or {},
            "added_at": datetime.utcnow().isoformat(),
            "word_count": len(content.split()),
        }
        
        logger.info(f"Added document {doc_id} ({len(content)} chars)")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search documents (basic keyword search).
        
        Future: Will use vector similarity search.
        """
        
        results = []
        query_lower = query.lower()
        
        for doc_id, doc in self.documents.items():
            content_lower = doc["content"].lower()
            
            # Simple keyword matching
            if query_lower in content_lower:
                # Calculate rough relevance score
                score = content_lower.count(query_lower) / len(content_lower.split())
                
                results.append({
                    "doc_id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score,
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document"""
        return self.documents.get(doc_id)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            logger.info(f"Deleted document {doc_id}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get document store statistics"""
        
        total_words = sum(doc["word_count"] for doc in self.documents.values())
        
        return {
            "total_documents": len(self.documents),
            "total_words": total_words,
            "embeddings_enabled": self.embeddings_available,
        }


class RAGTool:
    """
    RAG tool for knowledge-augmented responses.
    """
    
    def __init__(self):
        self.document_store = DocumentStore()
        self.rag_stats = {
            "total_queries": 0,
            "documents_retrieved": 0,
        }
    
    async def query(
        self,
        query: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Returns relevant documents and context for LLM.
        """
        
        logger.info(f"📚 RAG query: {query}")
        
        # Search documents
        results = self.document_store.search(query, top_k)
        
        # Update stats
        self.rag_stats["total_queries"] += 1
        self.rag_stats["documents_retrieved"] += len(results)
        
        # Format context for LLM
        context = self._format_context(results)
        
        return {
            "query": query,
            "results": results,
            "context": context,
            "found": len(results),
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context"""
        
        if not results:
            return "No relevant documents found in knowledge base."
        
        context = "# Relevant Information from Knowledge Base\n\n"
        
        for idx, result in enumerate(results, 1):
            context += f"## Source {idx}\n"
            context += f"{result['content']}\n\n"
            
            if result.get("metadata"):
                context += f"*Metadata: {result['metadata']}*\n\n"
            
            context += "---\n\n"
        
        return context
    
    async def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add new knowledge to the store"""
        
        import uuid
        doc_id = str(uuid.uuid4())
        
        self.document_store.add_document(doc_id, content, metadata)
        
        return doc_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG statistics"""
        
        return {
            **self.rag_stats,
            "store": self.document_store.get_stats(),
        }


# Global RAG tool instance
rag_tool = RAGTool()