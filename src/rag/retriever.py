"""
RAG Retriever
Showcases: Build RAG Applications (IBM)
"""
from typing import List, Dict
import logging
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

class RAGRetriever:
    """Retrieve relevant context for queries using RAG"""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize RAG retriever
        
        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store
    
    def retrieve_context(
        self, 
        query: str, 
        n_results: int = 5
    ) -> str:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            n_results: Number of results to retrieve
            
        Returns:
            Formatted context string
        """
        try:
            # Search vector store
            results = self.vector_store.search(query, n_results=n_results)
            
            if not results:
                return "No relevant context found."
            
            # Format context
            context = "Relevant Data Context:\n\n"
            for idx, result in enumerate(results, 1):
                context += f"[{idx}] {result['document']}\n\n"
            
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return "Error retrieving context."
    
    def retrieve_column_info(self, column_name: str) -> Dict:
        """
        Retrieve information about a specific column
        
        Args:
            column_name: Name of the column
            
        Returns:
            Column information dictionary
        """
        try:
            query = f"Information about column {column_name}"
            results = self.vector_store.search(query, n_results=3)
            
            for result in results:
                if result['metadata'].get('column_name') == column_name:
                    return {
                        "column_name": column_name,
                        "info": result['document'],
                        "metadata": result['metadata']
                    }
            
            return {"error": f"Column '{column_name}' not found"}
            
        except Exception as e:
            logger.error(f"Error retrieving column info: {str(e)}")
            return {"error": str(e)}

