"""
AI-Powered Business Intelligence Agent - Streamlit Version
Showcases Coursera Certifications:
- Build RAG Applications: Get Started (IBM)
- Vector Databases for RAG (IBM)
- Fundamentals of Building AI Agents (IBM)
- Python for Data Science, AI & Development (IBM)
- Statistics Foundations (Meta)
- Data Analysis with Spreadsheets and SQL (Meta)
- Exploratory Data Analysis for Machine Learning (IBM)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import logging
import sys

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

# Header
st.title("🤖 AI-Powered Business Intelligence Agent")
st.markdown("Upload your data and ask questions in natural language. The AI agent will analyze your data and provide insights.")

st.markdown("""
**🎓 Powered by Coursera Certifications:**
- IBM: RAG Applications, Vector Databases, AI Agents, Python, EDA
- Meta: Statistics, Data Analysis, SQL
""")

# Sidebar
with st.sidebar:
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls', 'parquet'],
        help="Upload your dataset (CSV, Excel, or Parquet)"
    )
    
    if uploaded_file is not None and st.button("Load Data", type="primary"):
        try:
            with st.spinner("Loading and processing data..."):
                # Load file
                loader = DataLoader(max_size_mb=settings.MAX_FILE_SIZE_MB)
                df, error = loader.load_file(uploaded_file)
                
                if error:
                    st.error(f"❌ Error: {error}")
                else:
                    # Process data
                    processor = DataProcessor()
                    cleaned_df = processor.clean_data(df)
                    
                    # Store in session state
                    st.session_state.df = cleaned_df
                    
                    # Initialize vector store
                    st.session_state.vector_store = VectorStore(
                        persist_directory=str(settings.CHROMA_DIR),
                        collection_name=settings.CHROMA_COLLECTION,
                        embedding_model=settings.EMBEDDING_MODEL
                    )
                    
                    # Clear existing data and add new
                    st.session_state.vector_store.clear()
                    st.session_state.vector_store.add_dataframe_context(cleaned_df)
                    
                    # Initialize agent
                    retriever = RAGRetriever(st.session_state.vector_store)
                    st.session_state.agent = BusinessIntelligenceAgent(
                        df=cleaned_df,
                        retriever=retriever,
                        llm_model=settings.LLM_MODEL,
                        api_key=settings.GEMINI_API_KEY,
                        temperature=settings.LLM_TEMPERATURE
                    )
                    
                    # Get summary
                    summary = loader.get_data_summary(cleaned_df)
                    quality = processor.get_data_quality_report(cleaned_df)
                    
                    st.success("✅ Data loaded successfully!")
                    st.metric("Rows", f"{summary['rows']:,}")
                    st.metric("Columns", summary['columns'])
                    st.metric("Missing %", f"{quality['missing_percentage']:.2f}%")
                    
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Error loading data: {str(e)}")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask Questions", "📊 Data Preview", "📈 Visualizations", "ℹ️ About"])

with tab1:
    st.header("Ask Questions About Your Data")
    
    if st.session_state.df is None:
        st.warning("⚠️ Please upload data first using the sidebar.")
    else:
        st.markdown("""
        **Example Questions:**
        - What are the summary statistics for [column_name]?
        - Is there a correlation between [var1] and [var2]?
        - Compare [value_column] across [group_column]
        - What factors are most important for predicting [target]?
        - Show me the top 10 values in [column]
        """)
        
        question = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the top 5 products by revenue?",
            height=100
        )
        
        if st.button("Ask AI Agent", type="primary"):
            if question.strip():
                with st.spinner("🤖 AI Agent is analyzing..."):
                    try:
                        result = st.session_state.agent.query(question)
                        
                        if result['success']:
                            st.markdown("### Answer:")
                            st.markdown(result['answer'])
                        else:
                            st.error(f"❌ {result['answer']}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("Please enter a question.")

with tab2:
    st.header("Data Preview")
    
    if st.session_state.df is not None:
        st.dataframe(st.session_state.df.head(50), use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(st.session_state.df))
        with col2:
            st.metric("Total Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("Memory (MB)", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    else:
        st.info("Upload data to see preview")

with tab3:
    st.header("Create Visualizations")
    
    if st.session_state.df is not None:
        viz_type = st.selectbox(
            "Visualization Type",
            ["Distribution", "Correlation Heatmap", "Scatter Plot", "Box Plot"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            column1 = st.selectbox("Select Column 1", st.session_state.df.columns.tolist())
        with col2:
            column2 = st.selectbox("Select Column 2 (for scatter)", st.session_state.df.columns.tolist())
        
        if st.button("Create Visualization", type="primary"):
            try:
                viz = st.session_state.visualizer
                
                if viz_type == "Distribution":
                    fig = viz.plot_distribution(st.session_state.df, column1)
                elif viz_type == "Correlation Heatmap":
                    fig = viz.plot_correlation_heatmap(st.session_state.df)
                elif viz_type == "Scatter Plot":
                    fig = viz.plot_scatter(st.session_state.df, column1, column2)
                elif viz_type == "Box Plot":
                    fig = viz.plot_box_plot(st.session_state.df, column1)
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error creating visualization: {str(e)}")
    else:
        st.info("Upload data to create visualizations")

with tab4:
    st.header("About This Project")
    
    st.markdown("""
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
    
    ### 🛠️ Premium Tech Stack
    
    **AI & LLM:**
    - 🤖 **Google Gemini** (LLM - FREE tier)
    - 🧠 **LangChain** (RAG orchestration & agent framework)
    - 🔗 **Chain-of-Thought Reasoning** (Complex query handling)
    
    **Vector & Semantic Search:**
    - 🎯 **ChromaDB** (Vector database - local, persistent)
    - 📊 **Sentence Transformers** (Embeddings - FREE)
    - 🔍 **Semantic Search** (RAG retrieval engine)
    
    **Data Science & Analytics:**
    - 🐼 **Pandas** (Data manipulation & processing)
    - 📈 **Plotly** (Interactive visualizations)
    - 🔬 **SciPy** (Statistical testing)
    - 📉 **NumPy** (Numerical computing)
    - 📊 **DuckDB** (SQL analytics)
    
    **UI & Deployment:**
    - 🎨 **Streamlit** (Interactive web interface)
    - ☁️ **Streamlit Cloud** (Serverless deployment)
    - 🎭 **Custom CSS/HTML** (Responsive design)
    
    ### 🚀 Advanced Features
    - 💬 **Multi-turn Conversations** - Remember context across questions
    - 🤖 **AI Agent System** - Automatically selects best analysis tools
    - 📚 **RAG Pipeline** - Retrieves relevant data context for answers
    - 🔬 **Statistical Analysis** - Hypothesis testing, correlation, regression
    - 📊 **Interactive Visualizations** - Distribution, heatmap, scatter, box plots
    - 🎯 **Natural Language to SQL** - Convert questions to queries
    - 🧹 **Auto Data Cleaning** - Handle missing values and anomalies
    - 💾 **Persistent Vector Store** - Reuse embeddings across sessions
    
    ---
    **Author:** Anix Lynch | [Portfolio](https://gozeroshot.dev) | [GitHub](https://github.com/anixlynch)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ showcasing Coursera certifications from IBM and Meta")

