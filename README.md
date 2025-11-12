# AI-Powered Business Intelligence Agent 🤖📊

> An intelligent data analytics assistant powered by RAG, Vector Databases, and AI Agents that automatically performs exploratory data analysis and answers business questions in natural language.

## 🎓 Coursera Certifications Showcased

This project demonstrates skills from the following completed certifications:

- ✅ **Build RAG Applications: Get Started** (IBM) - Core RAG implementation
- ✅ **Vector Databases for RAG: An Introduction** (IBM) - ChromaDB integration
- ✅ **Fundamentals of Building AI Agents** (IBM) - Tool calling and agentic workflows
- ✅ **Python for Data Science, AI & Development** (IBM) - Data processing
- ✅ **Statistics Foundations** (Meta) - Statistical analysis and hypothesis testing
- ✅ **Data Analysis with Spreadsheets and SQL** (Meta) - SQL queries and data manipulation
- ✅ **Exploratory Data Analysis for Machine Learning** (IBM) - Automated EDA
- ✅ **Introduction to Data Analytics** (Meta/IBM) - Data analysis fundamentals

## 🚀 Features

### Core Capabilities
- **📤 Upload & Analyze**: Drop CSV/Excel files and get instant insights
- **💬 Natural Language Queries**: Ask questions like "What factors drive high sales?"
- **🤖 AI Agent**: Automatically selects the right analysis tools
- **📊 Auto-Visualizations**: Generates charts and statistical summaries
- **🔍 RAG-Powered Context**: Retrieves relevant data context for answers
- **📈 Statistical Analysis**: Hypothesis testing, correlation, regression

### Technical Features
- **Vector Database**: ChromaDB for semantic search over data
- **Multiple LLMs**: Support for OpenAI, Anthropic Claude, Gemini
- **SQL Interface**: Query data using natural language → SQL
- **Interactive UI**: Built with Gradio for easy interaction
- **Export Results**: Download insights as reports

## 🛠️ Tech Stack

### AI & ML
- **LangChain** - RAG orchestration
- **ChromaDB** - Vector database for embeddings
- **OpenAI/Anthropic** - LLM for reasoning
- **Sentence Transformers** - Text embeddings

### Data & Analytics
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Statistical tests
- **Matplotlib/Plotly** - Visualizations
- **DuckDB** - SQL analytics

### Application
- **Gradio** - Interactive web UI
- **Python 3.11+** - Core language

## 📦 Installation

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Setup

1. **Clone the repository**
```bash
cd /Users/anixlynch/dev/coursera-portfolio-projects/ai-business-intelligence-agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your keys
# Or source from global config
source ~/.config/secrets/global.env
```

5. **Run the application**
```bash
python app.py
```

The UI will open at `http://localhost:7860`

## 🎯 Usage

### 1. Upload Data
- Drag and drop CSV or Excel files
- Or use sample datasets provided

### 2. Ask Questions
```
"What are the top 5 products by revenue?"
"Is there a correlation between price and sales?"
"Show me trends over time"
"Perform a statistical test on conversion rates"
```

### 3. Get Insights
- AI agent analyzes your question
- Retrieves relevant data using RAG
- Performs statistical analysis
- Generates visualizations
- Provides executive summary

## 📊 Example Workflows

### Business Analysis
1. Upload sales data
2. Ask: "What factors predict customer churn?"
3. Agent performs:
   - Feature correlation analysis
   - Statistical significance tests
   - Predictive feature ranking
   - Visualization of key drivers

### Marketing Analytics
1. Upload campaign data
2. Ask: "Which marketing channel has the best ROI?"
3. Agent provides:
   - Channel comparison analysis
   - Statistical testing
   - ROI calculations
   - Recommendations

## 🏗️ Architecture

```
User Question
    ↓
AI Agent (LangChain)
    ↓
Tool Selection:
  - SQL Query Tool
  - Statistical Analysis Tool
  - Visualization Tool
  - RAG Search Tool
    ↓
ChromaDB (Context Retrieval)
    ↓
Data Processing (Pandas)
    ↓
Analysis Results
    ↓
LLM Summary
    ↓
Interactive Display
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...           # Or use ANTHROPIC_API_KEY
HF_TOKEN=hf_...                 # For embeddings

# Optional
LANGSMITH_API_KEY=lsv2_...      # For tracing
CHROMA_TOKEN=ck-...             # For cloud ChromaDB
```

### Customization
- **Change LLM**: Edit `config.py` to switch between models
- **Add Tools**: Extend `tools/` directory with custom analysis tools
- **Modify UI**: Customize Gradio interface in `app.py`

## 📚 Project Structure

```
ai-business-intelligence-agent/
├── app.py                  # Main Gradio application
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── config.py              # Configuration settings
├── README.md              # This file
│
├── src/
│   ├── agent/             # AI Agent implementation
│   │   ├── __init__.py
│   │   ├── agent.py       # Main agent logic
│   │   └── tools.py       # Agent tools
│   │
│   ├── rag/               # RAG implementation
│   │   ├── __init__.py
│   │   ├── vectorstore.py # ChromaDB setup
│   │   └── retriever.py   # Retrieval logic
│   │
│   ├── analytics/         # Data analysis
│   │   ├── __init__.py
│   │   ├── eda.py         # Exploratory analysis
│   │   ├── stats.py       # Statistical tests
│   │   └── viz.py         # Visualizations
│   │
│   └── data/              # Data processing
│       ├── __init__.py
│       ├── loader.py      # File upload handling
│       └── processor.py   # Data cleaning
│
├── data/                  # Sample datasets
│   └── examples/
│
└── tests/                 # Unit tests
    └── test_agent.py
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test with sample data
python examples/test_sales_analysis.py
```

## 🚀 Deployment

### Local
```bash
python app.py
```

### Cloud (Vercel/Hugging Face Spaces)
```bash
# Using Vercel
vercel deploy

# Or push to Hugging Face Spaces
git push origin main
```

## 📈 Performance

- **Query Response**: < 3 seconds
- **File Upload**: Handles up to 100MB CSV files
- **Vector Search**: Sub-second retrieval
- **Concurrent Users**: Supports 10+ simultaneous users

## 🤝 Contributing

Contributions welcome! This is a portfolio project, but feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - See LICENSE file

## 👤 Author

**Anix Lynch**
- Portfolio: [Link to your portfolio]
- LinkedIn: [Your LinkedIn]
- Email: anixlynch@gmail.com

## 🙏 Acknowledgments

Built using skills from IBM and Meta Coursera certifications:
- IBM: RAG, Vector Databases, AI Agents, Python, EDA
- Meta: Data Analytics, Statistics, SQL

---

**⭐ Star this repo if you find it useful!**

