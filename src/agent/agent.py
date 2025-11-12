"""
Business Intelligence Agent
Showcases: Fundamentals of Building AI Agents (IBM), Build RAG Applications (IBM)
"""
import pandas as pd
from typing import Dict, Any, Optional
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
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
    
    def _create_tools(self) -> list:
        """Create tools for the agent"""
        tools = [
            Tool(
                name="get_column_info",
                func=self.analysis_tools.get_column_info,
                description="Get detailed information about a specific column in the dataset. Input should be the column name."
            ),
            Tool(
                name="calculate_statistics",
                func=self.analysis_tools.calculate_statistics,
                description="Calculate summary statistics for a column. Input should be the column name."
            ),
            Tool(
                name="test_correlation",
                func=lambda x: self.analysis_tools.test_correlation(*x.split(",")),
                description="Test correlation between two variables. Input should be 'variable1,variable2' (comma-separated)."
            ),
            Tool(
                name="compare_groups",
                func=lambda x: self.analysis_tools.compare_groups(*x.split(",")),
                description="Compare means across groups using statistical tests. Input should be 'group_column,value_column' (comma-separated)."
            ),
            Tool(
                name="perform_regression",
                func=lambda x: self.analysis_tools.perform_regression(*x.split(",")),
                description="Perform simple linear regression. Input should be 'x_column,y_column' (comma-separated)."
            ),
            Tool(
                name="get_top_values",
                func=self.analysis_tools.get_top_values,
                description="Get top N most frequent values in a column. Input should be the column name."
            ),
            Tool(
                name="get_feature_importance",
                func=self.analysis_tools.get_feature_importance,
                description="Calculate feature importance for a target variable. Input should be the target column name."
            ),
            Tool(
                name="retrieve_context",
                func=self.retriever.retrieve_context,
                description="Retrieve relevant context from the data using semantic search. Input should be a query about the data."
            )
        ]
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor"""
        
        # Create prompt template
        template = """You are a Business Intelligence AI Agent specialized in data analysis.
You have access to tools that can help you analyze datasets and answer business questions.

You have access to the following tools:
{tools}

Tool Names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Guidelines:
1. Always use retrieve_context first to understand the data structure
2. Use appropriate tools for statistical analysis
3. Provide clear, business-focused insights
4. If a column doesn't exist, suggest similar columns
5. Always interpret statistical results in business terms

Question: {input}
Thought: {agent_scratchpad}
"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        return agent_executor
    
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
            
            # Run agent
            result = self.agent.invoke({"input": question})
            
            return {
                "question": question,
                "answer": result.get("output", "No answer generated"),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": f"Error: {str(e)}",
                "success": False
            }
    
    def get_dataset_summary(self) -> str:
        """Get a summary of the dataset"""
        summary = f"Dataset Overview:\n\n"
        summary += f"Rows: {len(self.df)}\n"
        summary += f"Columns: {len(self.df.columns)}\n"
        summary += f"Column names: {', '.join(self.df.columns)}\n\n"
        summary += f"Sample data:\n{self.df.head(3).to_string()}\n"
        
        return summary

