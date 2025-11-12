# 🚀 Quick Start Guide

Get the AI Business Intelligence Agent running in 5 minutes!

## Prerequisites

- Python 3.11+
- pip or uv

## Step 1: Setup Environment

```bash
# Navigate to project
cd /Users/anixlynch/dev/coursera-portfolio-projects/ai-business-intelligence-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure API Keys

Your keys are already in `~/.config/secrets/global.env`. Just source them:

```bash
# Source global env (Mac/Linux)
source ~/.config/secrets/global.env

# Or create local .env file
cat > .env << 'EOF'
GEMINI_API_KEY=$GEMINI_API_KEY
HF_TOKEN=$HF_TOKEN
EOF
```

## Step 3: Run the App

```bash
python app.py
```

The app will open at: **http://localhost:7860**

## Step 4: Test It!

1. **Upload Data**
   - Go to "Upload Data" tab
   - Upload a CSV/Excel file
   - Wait for processing

2. **Ask Questions**
   - Go to "Ask Questions" tab
   - Try: "What are the summary statistics?"
   - Try: "What columns are available?"
   - Try: "Show me correlation between [col1] and [col2]"

3. **Create Visualizations**
   - Go to "Visualizations" tab
   - Select a visualization type
   - Choose columns
   - Click "Create Visualization"

## 📊 Sample Data

Want to test without your own data? Download sample datasets:

```bash
# Download sales sample
curl -o data/examples/sales_sample.csv https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv
```

## 🐛 Troubleshooting

### "No module named 'src'"
```bash
# Make sure you're in the project root
pwd  # Should end with: /ai-business-intelligence-agent
```

### "GEMINI_API_KEY not found"
```bash
# Check if env vars are loaded
echo $GEMINI_API_KEY

# If empty, source again
source ~/.config/secrets/global.env
```

### "ChromaDB connection error"
```bash
# ChromaDB will create local directory automatically
# Check if chroma_db/ folder exists
ls -la chroma_db/
```

## 🚀 Next Steps

- Deploy to HuggingFace Spaces (see `deployment/README_HF_SPACES.md`)
- Deploy to gozeroshot.dev
- Try with your own business data!

## 💡 Tips

- **FREE Tier**: Uses Google Gemini (60 req/min free)
- **Local Storage**: ChromaDB runs locally (no cloud costs)
- **Privacy**: All data processing happens locally
- **Upgrade**: Switch to OpenAI by changing one line in `config.py`

---

**Need help?** Check the full README.md or open an issue!

