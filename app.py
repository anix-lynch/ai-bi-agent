"""
AI-Powered Business Intelligence Agent
Main Gradio Application

Showcases Coursera Certifications:
- Build RAG Applications: Get Started (IBM)
- Vector Databases for RAG (IBM)
- Fundamentals of Building AI Agents (IBM)
- Python for Data Science, AI & Development (IBM)
- Statistics Foundations (Meta)
- Data Analysis with Spreadsheets and SQL (Meta)
- Exploratory Data Analysis for Machine Learning (IBM)
"""

import gradio as gr
import pandas as pd
from pathlib import Path
import logging
import sys
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.analytics.eda import EDAAnalyzer
from src.analytics.viz import Visualizer
from src.rag.vectorstore import VectorStore
from src.rag.retriever import RAGRetriever
from src.agent.agent import BusinessIntelligenceAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG_MODE else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
class AppState:
    """Application state"""
    df: Optional[pd.DataFrame] = None
    agent: Optional[BusinessIntelligenceAgent] = None
    vector_store: Optional[VectorStore] = None
    visualizer: Visualizer = Visualizer()
    
app_state = AppState()

def load_data(file) -> Tuple[str, str]:
    """
    Load uploaded data file
    
    Args:
        file: Uploaded file
        
    Returns:
        Tuple of (status_message, data_preview)
    """
    try:
        if file is None:
            return "❌ No file uploaded", ""
        
        # Load file
        loader = DataLoader(max_size_mb=settings.MAX_FILE_SIZE_MB)
        df, error = loader.load_file(file.name)
        
        if error:
            return f"❌ Error: {error}", ""
        
        # Process data
        processor = DataProcessor()
        cleaned_df = processor.clean_data(df)
        
        # Store in state
        app_state.df = cleaned_df
        
        # Initialize vector store
        app_state.vector_store = VectorStore(
            persist_directory=str(settings.CHROMA_DIR),
            collection_name=settings.CHROMA_COLLECTION,
            embedding_model=settings.EMBEDDING_MODEL
        )
        
        # Clear existing data and add new
        app_state.vector_store.clear()
        app_state.vector_store.add_dataframe_context(cleaned_df)
        
        # Initialize agent
        retriever = RAGRetriever(app_state.vector_store)
        app_state.agent = BusinessIntelligenceAgent(
            df=cleaned_df,
            retriever=retriever,
            llm_model=settings.LLM_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=settings.LLM_TEMPERATURE
        )
        
        # Get summary
        summary = loader.get_data_summary(cleaned_df)
        quality = processor.get_data_quality_report(cleaned_df)
        
        status = f"""✅ Data loaded successfully!

📊 **Dataset Summary:**
- Rows: {summary['rows']:,}
- Columns: {summary['columns']}
- Memory: {summary['memory_usage_mb']:.2f} MB

📋 **Columns:** {', '.join(summary['column_names'][:10])}{"..." if len(summary['column_names']) > 10 else ""}

🔍 **Data Quality:**
- Missing values: {quality['missing_percentage']:.2f}%
- Duplicate rows: {quality['duplicate_rows']}

🤖 **AI Agent Ready!** Ask questions about your data.
"""
        
        preview = cleaned_df.head(10).to_html()
        
        return status, preview
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return f"❌ Error: {str(e)}", ""

def ask_question(question: str) -> str:
    """
    Ask the AI agent a question
    
    Args:
        question: User question
        
    Returns:
        Agent's answer
    """
    try:
        if app_state.agent is None:
            return "❌ Please upload data first!"
        
        if not question.strip():
            return "❌ Please enter a question."
        
        # Query agent
        result = app_state.agent.query(question)
        
        if result['success']:
            return f"**Answer:**\n\n{result['answer']}"
        else:
            return f"❌ {result['answer']}"
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return f"❌ Error: {str(e)}"

def create_visualization(viz_type: str, column1: str, column2: str = None):
    """Create visualization"""
    try:
        if app_state.df is None:
            return None, "❌ Please upload data first!"
        
        if not column1:
            return None, "❌ Please select a column."
        
        viz = app_state.visualizer
        
        if viz_type == "Distribution":
            fig = viz.plot_distribution(app_state.df, column1)
        elif viz_type == "Correlation Heatmap":
            fig = viz.plot_correlation_heatmap(app_state.df)
        elif viz_type == "Scatter Plot":
            if not column2:
                return None, "❌ Please select a second column for scatter plot."
            fig = viz.plot_scatter(app_state.df, column1, column2)
        elif viz_type == "Box Plot":
            fig = viz.plot_box_plot(app_state.df, column1)
        else:
            return None, "❌ Invalid visualization type."
        
        return fig, "✅ Visualization created!"
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None, f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="AI Business Intelligence Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI-Powered Business Intelligence Agent
    
    Upload your data and ask questions in natural language. The AI agent will analyze your data and provide insights.
    
    **🎓 Powered by Coursera Certifications:**
    - IBM: RAG Applications, Vector Databases, AI Agents, Python, EDA
    - Meta: Statistics, Data Analysis, SQL
    """)
    
    with gr.Tab("📤 Upload Data"):
        gr.Markdown("### Upload your dataset (CSV, Excel, or Parquet)")
        
        with gr.Row():
            file_input = gr.File(
                label="Upload Data File",
                file_types=[".csv", ".xlsx", ".xls", ".parquet"]
            )
        
        upload_btn = gr.Button("Load Data", variant="primary", size="lg")
        status_output = gr.Markdown()
        preview_output = gr.HTML(label="Data Preview")
        
        upload_btn.click(
            fn=load_data,
            inputs=[file_input],
            outputs=[status_output, preview_output]
        )
    
    with gr.Tab("💬 Ask Questions"):
        gr.Markdown("### Ask questions about your data in natural language")
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the top 5 products by revenue? Is there a correlation between price and sales?",
                lines=3
            )
        
        ask_btn = gr.Button("Ask AI Agent", variant="primary", size="lg")
        answer_output = gr.Markdown()
        
        gr.Markdown("""
        **Example Questions:**
        - What are the summary statistics for [column_name]?
        - Is there a correlation between [var1] and [var2]?
        - Compare [value_column] across [group_column]
        - What factors are most important for predicting [target]?
        - Show me the top 10 values in [column]
        """)
        
        ask_btn.click(
            fn=ask_question,
            inputs=[question_input],
            outputs=[answer_output]
        )
    
    with gr.Tab("📊 Visualizations"):
        gr.Markdown("### Create visualizations")
        gr.Markdown("⚠️ Upload data first, then manually type the column name")
        
        with gr.Row():
            viz_type = gr.Dropdown(
                label="Visualization Type",
                choices=["Distribution", "Correlation Heatmap", "Scatter Plot", "Box Plot"],
                value="Distribution"
            )
        
        with gr.Row():
            col1 = gr.Textbox(label="Column 1", placeholder="Enter column name")
            col2 = gr.Textbox(label="Column 2 (for scatter)", placeholder="Enter column name")
        
        viz_btn = gr.Button("Create Visualization", variant="primary")
        viz_output = gr.Plot()
        viz_status = gr.Markdown()
        
        viz_btn.click(
            fn=create_visualization,
            inputs=[viz_type, col1, col2],
            outputs=[viz_output, viz_status]
        )
    
    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## About This Project
        
        This AI-Powered Business Intelligence Agent showcases skills from multiple Coursera certifications:
        
        ### 🎓 IBM Certifications
        - **Build RAG Applications: Get Started** - RAG implementation with LangChain
        - **Vector Databases for RAG** - ChromaDB for semantic search
        - **Fundamentals of Building AI Agents** - Agent with tool calling
        - **Python for Data Science, AI & Development** - Data processing
        - **Exploratory Data Analysis for Machine Learning** - Automated EDA
        
        ### 🎓 Meta Certifications
        - **Statistics Foundations** - Statistical testing and analysis
        - **Data Analysis with Spreadsheets and SQL** - Data manipulation
        - **Introduction to Data Analytics** - Analytics fundamentals
        
        ### 🛠️ Tech Stack
        - **LLM:** Google Gemini (FREE tier)
        - **Vector DB:** ChromaDB (local)
        - **Framework:** LangChain
        - **Embeddings:** Sentence Transformers (FREE)
        - **UI:** Gradio
        
        ### 🚀 Features
        - Natural language queries
        - Automated statistical analysis
        - RAG-powered context retrieval
        - Interactive visualizations
        - AI agent with multiple tools
        
        ---
        **Author:** Anix Lynch | [Portfolio](https://gozeroshot.dev) | [GitHub](https://github.com/anixlynch)
        """)

if __name__ == "__main__":
    logger.info("Starting AI Business Intelligence Agent...")
    logger.info(f"Using LLM: {settings.LLM_MODEL}")
    
    # Launch with HuggingFace Spaces compatible settings
    demo.launch()

