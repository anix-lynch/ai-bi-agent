"""
Data loader for handling file uploads
Showcases: Data Analysis with Spreadsheets and SQL (Meta)
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Handle loading various data file formats"""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.parquet', '.json']
    
    def __init__(self, max_size_mb: int = 100):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
    
    def load_file(self, file_path_or_object) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load data from file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Tuple of (DataFrame, error_message)
        """
        try:
            path = Path(file_path_or_object)
            
            # Check file exists
            if not path.exists():
                return None, f"File not found: {file_path_or_object}"
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_size_bytes:
                return None, f"File too large. Maximum size: {self.max_size_mb}MB"
            
            # Check file extension
            ext = path.suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                return None, f"Unsupported format. Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            
            # Load based on extension (handle both file path and UploadedFile)
            # For UploadedFile, pandas functions work with file-like objects
            if ext == '.csv':
                df = pd.read_csv(file_path_or_object)
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path_or_object)
            elif ext == '.parquet':
                df = pd.read_parquet(file_path_or_object)
            elif ext == '.json':
                df = pd.read_json(file_path_or_object)
            else:
                return None, f"Unsupported file type: {ext}"
            
            # Basic validation
            if df.empty:
                return None, "File is empty"
            
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
            return df, None
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return None, f"Error loading file: {str(e)}"
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the dataframe"""
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        }

