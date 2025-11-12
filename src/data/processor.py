"""
Data processor for cleaning and transforming data
Showcases: EDA for Machine Learning (IBM), Statistics Foundations (Meta)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and clean data for analysis"""
    
    def __init__(self):
        self.original_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning
        
        Args:
            df: Input dataframe
            
        Returns:
            Cleaned dataframe
        """
        self.original_df = df.copy()
        cleaned_df = df.copy()
        
        # Remove duplicate rows
        initial_rows = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_df)
        
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values strategy
        # For numeric columns: fill with median
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_df[col].isnull().any():
                median_val = cleaned_df[col].median()
                cleaned_df[col].fillna(median_val, inplace=True)
        
        # For categorical columns: fill with mode or 'Unknown'
        categorical_cols = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                mode_val = cleaned_df[col].mode()
                if len(mode_val) > 0:
                    cleaned_df[col].fillna(mode_val[0], inplace=True)
                else:
                    cleaned_df[col].fillna('Unknown', inplace=True)
        
        self.processed_df = cleaned_df
        return cleaned_df
    
    def get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of numeric columns"""
        return df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of categorical columns"""
        return df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of datetime columns"""
        return df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detect and categorize column types"""
        return {
            "numeric": self.get_numeric_columns(df),
            "categorical": self.get_categorical_columns(df),
            "datetime": self.get_datetime_columns(df)
        }
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate data quality report"""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_percentage": (missing_cells / total_cells * 100) if total_cells > 0 else 0,
            "duplicate_rows": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_types": self.detect_column_types(df)
        }

