"""
Exploratory Data Analysis
Showcases: Exploratory Data Analysis for Machine Learning (IBM)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """Perform exploratory data analysis"""
    
    def __init__(self):
        pass
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary of summary statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            "shape": df.shape,
            "numeric_summary": df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
            "categorical_summary": self._get_categorical_summary(df),
            "correlations": self._get_correlations(df),
            "outliers": self._detect_outliers(df)
        }
        
        return summary
    
    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of categorical columns"""
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        summary = {}
        
        for col in cat_cols:
            value_counts = df[col].value_counts()
            summary[col] = {
                "unique_values": df[col].nunique(),
                "top_5": value_counts.head(5).to_dict(),
                "missing": df[col].isnull().sum()
            }
        
        return summary
    
    def _get_correlations(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """Get significant correlations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # Find significant correlations
        significant_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    significant_corrs.append({
                        "var1": corr_matrix.columns[i],
                        "var2": corr_matrix.columns[j],
                        "correlation": round(corr_val, 3)
                    })
        
        return {
            "matrix": corr_matrix.to_dict(),
            "significant": significant_corrs
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outlier_count > 0:
                outliers[col] = {
                    "count": int(outlier_count),
                    "percentage": round(outlier_count / len(df) * 100, 2),
                    "lower_bound": round(lower_bound, 2),
                    "upper_bound": round(upper_bound, 2)
                }
        
        return outliers
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str) -> List[Dict]:
        """
        Calculate feature importance based on correlation with target
        
        Args:
            df: Input dataframe
            target_col: Target column name
            
        Returns:
            List of features with importance scores
        """
        if target_col not in df.columns:
            return []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]
        
        if len(numeric_cols) == 0:
            return []
        
        correlations = df[numeric_cols + [target_col]].corr()[target_col].drop(target_col)
        
        feature_importance = [
            {
                "feature": col,
                "correlation": round(abs(correlations[col]), 3),
                "direction": "positive" if correlations[col] > 0 else "negative"
            }
            for col in correlations.index
        ]
        
        feature_importance.sort(key=lambda x: x["correlation"], reverse=True)
        
        return feature_importance

