"""
Configuration settings for AI Business Intelligence Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings - simple class without Pydantic to avoid Gradio type inference issues"""
    
    def __init__(self):
        # Project paths
        self.PROJECT_ROOT = Path(__file__).parent
        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.UPLOAD_DIR = self.DATA_DIR / "uploads"
        self.CACHE_DIR = self.DATA_DIR / "cache"
        self.CHROMA_DIR = self.PROJECT_ROOT / "chroma_db"
        
        # API Keys
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        self.HF_TOKEN = os.getenv("HF_TOKEN", "")
        
        # LLM Configuration
        self.LLM_PROVIDER = "gemini"
        self.LLM_MODEL = "gemini-1.5-flash"
        self.LLM_TEMPERATURE = 0.1
        
        # Embedding Configuration
        self.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        self.EMBEDDING_DIMENSION = 384
        
        # ChromaDB Configuration
        self.CHROMA_COLLECTION = "business_data"
        
        # Application Configuration
        self.APP_PORT = int(os.getenv("APP_PORT", "7860"))
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.MAX_FILE_SIZE_MB = 100
        
        # Agent Configuration
        self.MAX_ITERATIONS = 10
        self.AGENT_VERBOSE = True
        
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

