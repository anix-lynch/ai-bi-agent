import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from src.data.loader import DataLoader
from src.data.processor import DataProcessor
from src.analytics.viz import Visualizer
from src.rag.vectorstore import VectorStore
from src.rag.retriever import RAGRetriever
from src.agent.agent import BusinessIntelligenceAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Business Intelligence Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n, freq='D'),
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Headphones', 'Watch'], n),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'Sales': np.random.randint(500, 5000, n),
        'Quantity': np.random.randint(1, 50, n),
        'Customer_ID': np.random.randint(1000, 2000, n),
        'Discount': np.random.uniform(0, 0.3, n),
        'Customer_Age': np.random.randint(18, 75, n),
        'Satisfaction': np.random.randint(1, 5, n)
    })
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Profit_Margin'] = (df['Sales'] * (1 - df['Discount'])) * 0.4
    return df

if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

st.title("🤖 AI-Powered Business Intelligence Agent")
st.markdown("**Click demo below for instant visualizations & AI insights**")

with st.sidebar:
    st.header("📊 Load Data")
    
    if st.button("🎯 Load Demo (500 Records)", type="primary", use_container_width=True):
        with st.spinner("Loading demo data..."):
            try:
                demo_df = generate_demo_data()
                processor = DataProcessor()
                cleaned_df = processor.clean_data(demo_df)
                
                st.session_state.df = cleaned_df
                
                st.session_state.vector_store = VectorStore(
                    persist_directory=str(settings.CHROMA_DIR),
                    collection_name=settings.CHROMA_COLLECTION,
                    embedding_model=settings.EMBEDDING_MODEL
                )
                st.session_state.vector_store.clear()
                st.session_state.vector_store.add_dataframe_context(cleaned_df)
                
                retriever = RAGRetriever(st.session_state.vector_store)
                st.session_state.agent = BusinessIntelligenceAgent(
                    df=cleaned_df,
                    retriever=retriever,
                    llm_model=settings.LLM_MODEL,
                    api_key=settings.GEMINI_API_KEY,
                    temperature=settings.LLM_TEMPERATURE
                )
                
                st.success("✅ Demo loaded!")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    st.divider()
    st.header("📤 Upload CSV")
    uploaded_file = st.file_uploader("Choose file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None and st.button("Load CSV", use_container_width=True):
        with st.spinner("Loading..."):
            try:
                loader = DataLoader(max_size_mb=settings.MAX_FILE_SIZE_MB)
                df, error = loader.load_file(uploaded_file)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    processor = DataProcessor()
                    cleaned_df = processor.clean_data(df)
                    st.session_state.df = cleaned_df
                    
                    st.session_state.vector_store = VectorStore(
                        persist_directory=str(settings.CHROMA_DIR),
                        collection_name=settings.CHROMA_COLLECTION,
                        embedding_model=settings.EMBEDDING_MODEL
                    )
                    st.session_state.vector_store.clear()
                    st.session_state.vector_store.add_dataframe_context(cleaned_df)
                    
                    retriever = RAGRetriever(st.session_state.vector_store)
                    st.session_state.agent = BusinessIntelligenceAgent(
                        df=cleaned_df,
                        retriever=retriever,
                        llm_model=settings.LLM_MODEL,
                        api_key=settings.GEMINI_API_KEY,
                        temperature=settings.LLM_TEMPERATURE
                    )
                    
                    st.success("✅ Data loaded!")
            except Exception as e:
                st.error(f"❌ {str(e)}")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask AI", "📊 Data", "📈 Heatmap", "ℹ️ About"])

with tab1:
    st.header("Ask Questions")
    
    if st.session_state.df is None:
        st.info("👈 Click 'Load Demo' in sidebar to start")
    else:
        question = st.text_area("Question:", placeholder="e.g., What's the top product by sales?", height=80)
        
        if st.button("Get Answer", type="primary"):
            if question.strip():
                with st.spinner("🤖 Thinking..."):
                    try:
                        result = st.session_state.agent.query(question)
                        if result['success']:
                            st.markdown(result['answer'])
                        else:
                            st.error(result['answer'])
                    except Exception as e:
                        st.error(f"❌ {str(e)}")

with tab2:
    st.header("Data Preview")
    
    if st.session_state.df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(st.session_state.df))
        with col2:
            st.metric("Columns", len(st.session_state.df.columns))
        with col3:
            st.metric("MB", f"{st.session_state.df.memory_usage(deep=True).sum() / 1024**2:.1f}")
        
        st.dataframe(st.session_state.df, use_container_width=True)
    else:
        st.info("Load data first")

with tab3:
    st.header("Correlation Heatmap")
    
    if st.session_state.df is not None:
        try:
            viz = st.session_state.visualizer
            fig = viz.plot_correlation_heatmap(st.session_state.df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not create heatmap")
        except Exception as e:
            st.error(f"❌ {str(e)}")
    else:
        st.info("Load data first")

with tab4:
    st.header("About")
    st.markdown("""
    **AI-Powered BI Agent** | RAG + ChromaDB + LangChain
    
    🛠️ **Stack:**
    - 🤖 Google Gemini
    - 🧠 LangChain + RAG
    - 🎯 ChromaDB Vector DB
    - 📊 Plotly Visualizations
    - 🐼 Pandas + SciPy
    
    **Author:** Anix Lynch | [GitHub](https://github.com/anixlynch) | [Portfolio](https://gozeroshot.dev)
    """)

st.markdown("---")
st.markdown("Built with ❤️")
