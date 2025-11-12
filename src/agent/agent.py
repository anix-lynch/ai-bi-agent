"""
Business Intelligence Agent
Showcases: Fundamentals of Building AI Agents (IBM), Build RAG Applications (IBM)
"""
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.agent.tools import AnalysisTools
from src.rag.retriever import RAGRetriever

logger = logging.getLogger(__name__)

class BusinessIntelligenceAgent:
    """AI Agent for business intelligence analysis"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        retriever: RAGRetriever,
        llm_model: str = "gemini-1.5-flash",
        api_key: str = None,
        temperature: float = 0.1
    ):
        """
        Initialize BI Agent
        
        Args:
            df: Dataframe to analyze
            retriever: RAG retriever for context
            llm_model: LLM model name
            api_key: API key for LLM
            temperature: LLM temperature
        """
        self.df = df
        self.retriever = retriever
        self.analysis_tools = AnalysisTools(df)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature
        )
        
        # Create agent tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_tools(self) -> Dict[str, Any]:
        """Create tools mapping for the agent"""
        return {
            "get_column_info": {
                "func": self.analysis_tools.get_column_info,
                "description": "Get information about a column"
            },
            "calculate_statistics": {
                "func": self.analysis_tools.calculate_statistics,
                "description": "Calculate statistics for a column"
            },
            "get_top_values": {
                "func": self.analysis_tools.get_top_values,
                "description": "Get most frequent values in a column"
            },
            "retrieve_context": {
                "func": self.retriever.retrieve_context,
                "description": "Search for relevant data context"
            }
        }
    
    def _create_agent(self):
        """Create the agent (simplified without deprecated patterns)"""
        # Just return the LLM - we'll use a simple tool-calling pattern
        return self.llm
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the agent
        
        Args:
            question: Business question to answer
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # Get context from RAG
            context = self.retriever.retrieve_context(question)
            
            # Build prompt with available tools and data context
            prompt = f"""You are a Business Intelligence AI analyzing a dataset.

Dataset Overview:
{self.get_dataset_summary()}

Retrieved Context:
{context}

Available Tools:
{self._format_tools()}

User Question: {question}

Provide a clear, data-driven answer using the dataset information. If specific analysis is needed, describe what you would do with the available tools."""

            # Get response from LLM
            response = self.agent.invoke(prompt)
            
            # Extract text from response
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "question": question,
                "answer": answer,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "success": False
            }
    
    def _format_tools(self) -> str:
        """Format tools description for prompt"""
        tools_desc = []
        for name, info in self.tools.items():
            tools_desc.append(f"- {name}: {info['description']}")
        return "\n".join(tools_desc)
    
    def get_dataset_summary(self) -> str:
        """Get a summary of the dataset"""
        summary = f"Dataset Overview:\n\n"
        summary += f"Rows: {len(self.df)}\n"
        summary += f"Columns: {len(self.df.columns)}\n"
        summary += f"Column names: {', '.join(self.df.columns)}\n\n"
        summary += f"Sample data:\n{self.df.head(3).to_string()}\n"
        
        return summary

