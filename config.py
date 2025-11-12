"""
Configuration settings for AI Business Intelligence Agent
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    CACHE_DIR: Path = DATA_DIR / "cache"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
    
    # API Keys
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    
    # LLM Configuration
    LLM_PROVIDER: str = "gemini"  # Options: gemini, openai, anthropic
    LLM_MODEL: str = "gemini-1.5-flash"  # Fast and free
    LLM_TEMPERATURE: float = 0.1
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, local, free
    EMBEDDING_DIMENSION: int = 384
    
    # ChromaDB Configuration
    CHROMA_COLLECTION: str = "business_data"
    
    # Application Configuration
    APP_PORT: int = int(os.getenv("APP_PORT", "7860"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    MAX_FILE_SIZE_MB: int = 100
    
    # Agent Configuration
    MAX_ITERATIONS: int = 10
    AGENT_VERBOSE: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()

