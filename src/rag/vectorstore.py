"""
Vector Store using ChromaDB
Showcases: Vector Databases for RAG (IBM)
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage ChromaDB vector store for data context"""
    
    def __init__(
        self, 
        persist_directory: str,
        collection_name: str = "business_data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
            embedding_model: HuggingFace embedding model
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Business data context"}
        )
        
        logger.info(f"ChromaDB initialized with collection: {collection_name}")
    
    def add_dataframe_context(
        self, 
        df: pd.DataFrame, 
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Add dataframe context to vector store
        
        Args:
            df: Input dataframe
            metadata: Additional metadata
            
        Returns:
            Number of documents added
        """
        try:
            documents = []
            metadatas = []
            ids = []
            
            # Add column information
            for idx, col in enumerate(df.columns):
                doc = f"Column: {col}\n"
                doc += f"Data type: {df[col].dtype}\n"
                doc += f"Non-null count: {df[col].count()}\n"
                
                # Add statistics for numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    doc += f"Mean: {df[col].mean():.2f}\n"
                    doc += f"Median: {df[col].median():.2f}\n"
                    doc += f"Min: {df[col].min():.2f}\n"
                    doc += f"Max: {df[col].max():.2f}\n"
                
                # Add unique values for categorical columns
                elif pd.api.types.is_object_dtype(df[col]):
                    unique_vals = df[col].unique()[:5]  # Top 5 unique values
                    doc += f"Sample values: {', '.join(map(str, unique_vals))}\n"
                
                documents.append(doc)
                metadatas.append({
                    "type": "column_info",
                    "column_name": col,
                    "data_type": str(df[col].dtype),
                    **(metadata or {})
                })
                ids.append(f"col_{idx}_{col}")
            
            # Add sample rows
            sample_size = min(10, len(df))
            for idx in range(sample_size):
                row = df.iloc[idx]
                doc = f"Sample row {idx + 1}:\n"
                doc += "\n".join([f"{col}: {val}" for col, val in row.items()])
                
                documents.append(doc)
                metadatas.append({
                    "type": "sample_row",
                    "row_index": idx,
                    **(metadata or {})
                })
                ids.append(f"row_{idx}")
            
            # Add summary statistics
            summary_doc = f"Dataset Summary:\n"
            summary_doc += f"Total rows: {len(df)}\n"
            summary_doc += f"Total columns: {len(df.columns)}\n"
            summary_doc += f"Columns: {', '.join(df.columns)}\n"
            
            documents.append(summary_doc)
            metadatas.append({
                "type": "summary",
                **(metadata or {})
            })
            ids.append("summary")
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Error adding dataframe context: {str(e)}")
            return 0
    
    def search(
        self, 
        query: str, 
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search vector store
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for idx in range(len(results['ids'][0])):
                formatted_results.append({
                    "id": results['ids'][0][idx],
                    "document": results['documents'][0][idx],
                    "metadata": results['metadatas'][0][idx],
                    "distance": results['distances'][0][idx] if 'distances' in results else None
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def clear(self):
        """Clear the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Business data context"}
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {}

