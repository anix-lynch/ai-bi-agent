# Deploy to Streamlit Cloud

## Quick Deploy (Recommended)

1. **Push to GitHub** (if not already done)
   ```bash
   git remote add origin https://github.com/yourusername/ai-bi-agent.git
   git push -u origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - Sign in with GitHub

3. **Deploy**
   - Click "New app"
   - Repository: `yourusername/ai-bi-agent`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Add Secrets**
   - In your app dashboard, go to "Settings" → "Secrets"
   - Add your API keys:
   ```toml
   GEMINI_API_KEY = "your-gemini-key-here"
   HF_TOKEN = "your-hf-token-here"
   ```

5. **Done!**
   - Your app will be live at: `https://yourusername-ai-bi-agent.streamlit.app`

## Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## Features

✅ No Gradio type inference issues
✅ Better state management with Streamlit
✅ Simpler deployment process
✅ Free hosting on Streamlit Cloud
✅ Auto-updates on git push
✅ Built-in secrets management

## Requirements

- GitHub account
- Streamlit Cloud account (free)
- Gemini API key (free tier available)

## Notes

- Streamlit Cloud provides 1GB RAM for free tier apps
- Apps auto-sleep after inactivity (wakes up instantly on access)
- Much more stable than HuggingFace Spaces for Python apps

