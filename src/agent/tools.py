"""
Agent Tools
Showcases: Fundamentals of Building AI Agents (IBM)
"""
import pandas as pd
from typing import Dict, Any, Optional
import logging
from src.analytics.eda import EDAAnalyzer
from src.analytics.stats import StatisticalAnalyzer
from src.analytics.viz import Visualizer

logger = logging.getLogger(__name__)

class AnalysisTools:
    """Tools for the AI agent to perform data analysis"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analysis tools
        
        Args:
            df: Dataframe to analyze
        """
        self.df = df
        self.eda_analyzer = EDAAnalyzer()
        self.stats_analyzer = StatisticalAnalyzer()
        self.visualizer = Visualizer()
    
    def get_column_info(self, column_name: str) -> str:
        """Get information about a specific column"""
        try:
            if column_name not in self.df.columns:
                return f"Column '{column_name}' not found. Available columns: {', '.join(self.df.columns)}"
            
            col = self.df[column_name]
            info = f"Column: {column_name}\n"
            info += f"Data type: {col.dtype}\n"
            info += f"Non-null count: {col.count()}\n"
            info += f"Missing values: {col.isnull().sum()}\n"
            
            if pd.api.types.is_numeric_dtype(col):
                info += f"\nStatistics:\n"
                info += f"Mean: {col.mean():.2f}\n"
                info += f"Median: {col.median():.2f}\n"
                info += f"Std Dev: {col.std():.2f}\n"
                info += f"Min: {col.min():.2f}\n"
                info += f"Max: {col.max():.2f}\n"
            else:
                info += f"\nUnique values: {col.nunique()}\n"
                top_values = col.value_counts().head(5)
                info += f"Top 5 values:\n{top_values.to_string()}\n"
            
            return info
            
        except Exception as e:
            return f"Error getting column info: {str(e)}"
    
    def calculate_statistics(self, column_name: str) -> str:
        """Calculate statistics for a column"""
        try:
            if column_name not in self.df.columns:
                return f"Column '{column_name}' not found"
            
            summary = self.eda_analyzer.get_summary_statistics(self.df[[column_name]])
            return str(summary)
            
        except Exception as e:
            return f"Error calculating statistics: {str(e)}"
    
    def test_correlation(self, var1: str, var2: str) -> str:
        """Test correlation between two variables"""
        try:
            result = self.stats_analyzer.correlation_test(self.df, var1, var2)
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            output = f"Correlation Analysis: {var1} vs {var2}\n\n"
            output += f"Pearson Correlation: {result['pearson']['correlation']:.4f}\n"
            output += f"P-value: {result['pearson']['p_value']:.6f}\n"
            output += f"Significant: {result['pearson']['significant']}\n"
            output += f"Strength: {result['pearson']['strength']}\n"
            
            return output
            
        except Exception as e:
            return f"Error testing correlation: {str(e)}"
    
    def compare_groups(self, group_col: str, value_col: str) -> str:
        """Compare means across groups"""
        try:
            result = self.stats_analyzer.hypothesis_test_means(
                self.df, group_col, value_col
            )
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            output = f"Group Comparison: {value_col} by {group_col}\n\n"
            output += f"Test: {result['test']}\n"
            output += f"Number of groups: {result['n_groups']}\n"
            output += f"P-value: {result['p_value']:.6f}\n"
            output += f"Significant: {result['significant']}\n\n"
            output += "Group Means:\n"
            for group, mean in result['group_means'].items():
                output += f"  {group}: {mean}\n"
            output += f"\nInterpretation: {result['interpretation']}\n"
            
            return output
            
        except Exception as e:
            return f"Error comparing groups: {str(e)}"
    
    def perform_regression(self, x_col: str, y_col: str) -> str:
        """Perform simple linear regression"""
        try:
            result = self.stats_analyzer.simple_regression(self.df, x_col, y_col)
            
            if "error" in result:
                return f"Error: {result['error']}"
            
            output = f"Linear Regression: {y_col} ~ {x_col}\n\n"
            output += f"Equation: {result['equation']}\n"
            output += f"R-squared: {result['statistics']['r_squared']:.4f}\n"
            output += f"P-value: {result['statistics']['p_value']:.6f}\n"
            output += f"Significant: {result['statistics']['significant']}\n\n"
            output += f"Interpretation: {result['interpretation']}\n"
            
            return output
            
        except Exception as e:
            return f"Error performing regression: {str(e)}"
    
    def get_top_values(self, column_name: str, n: int = 10) -> str:
        """Get top N values from a column"""
        try:
            if column_name not in self.df.columns:
                return f"Column '{column_name}' not found"
            
            top_values = self.df[column_name].value_counts().head(n)
            output = f"Top {n} values in {column_name}:\n\n"
            output += top_values.to_string()
            
            return output
            
        except Exception as e:
            return f"Error getting top values: {str(e)}"
    
    def filter_data(self, condition: str) -> str:
        """Filter data based on a condition"""
        try:
            # This is a simplified version - in production, use safe evaluation
            filtered_df = self.df.query(condition)
            
            output = f"Filtered data with condition: {condition}\n\n"
            output += f"Rows matching condition: {len(filtered_df)}\n"
            output += f"Percentage: {len(filtered_df)/len(self.df)*100:.2f}%\n\n"
            output += "Sample of filtered data:\n"
            output += filtered_df.head(5).to_string()
            
            return output
            
        except Exception as e:
            return f"Error filtering data: {str(e)}"
    
    def get_feature_importance(self, target_col: str) -> str:
        """Get feature importance for a target column"""
        try:
            importance = self.eda_analyzer.get_feature_importance(self.df, target_col)
            
            if not importance:
                return "No numeric features found for importance calculation"
            
            output = f"Feature Importance for {target_col}:\n\n"
            for item in importance[:10]:  # Top 10
                output += f"{item['feature']}: {item['correlation']:.4f} ({item['direction']})\n"
            
            return output
            
        except Exception as e:
            return f"Error calculating feature importance: {str(e)}"

