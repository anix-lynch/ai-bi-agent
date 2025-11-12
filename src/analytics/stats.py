"""
Statistical Analysis
Showcases: Statistics Foundations (Meta), Supervised ML: Regression (IBM)
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzer:
    """Perform statistical analyses"""
    
    def __init__(self):
        pass
    
    def hypothesis_test_means(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        value_col: str,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform t-test or ANOVA for comparing means
        
        Args:
            df: Input dataframe
            group_col: Column containing groups
            value_col: Column containing values to compare
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        try:
            groups = df[group_col].unique()
            
            if len(groups) < 2:
                return {"error": "Need at least 2 groups for comparison"}
            
            # Prepare group data
            group_data = [df[df[group_col] == group][value_col].dropna() for group in groups]
            
            if len(groups) == 2:
                # Two sample t-test
                stat, pvalue = stats.ttest_ind(group_data[0], group_data[1])
                test_name = "Independent T-Test"
            else:
                # One-way ANOVA
                stat, pvalue = stats.f_oneway(*group_data)
                test_name = "One-Way ANOVA"
            
            result = {
                "test": test_name,
                "groups": groups.tolist(),
                "n_groups": len(groups),
                "statistic": round(float(stat), 4),
                "p_value": round(float(pvalue), 6),
                "significant": pvalue < alpha,
                "alpha": alpha,
                "interpretation": self._interpret_hypothesis_test(pvalue, alpha)
            }
            
            # Add group means
            result["group_means"] = {
                str(group): round(float(data.mean()), 2) 
                for group, data in zip(groups, group_data)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in hypothesis test: {str(e)}")
            return {"error": str(e)}
    
    def correlation_test(
        self, 
        df: pd.DataFrame, 
        var1: str, 
        var2: str
    ) -> Dict:
        """
        Test correlation between two variables
        
        Args:
            df: Input dataframe
            var1: First variable
            var2: Second variable
            
        Returns:
            Correlation test results
        """
        try:
            # Remove missing values
            data = df[[var1, var2]].dropna()
            
            if len(data) < 3:
                return {"error": "Need at least 3 observations"}
            
            # Pearson correlation
            corr, pvalue = stats.pearsonr(data[var1], data[var2])
            
            # Spearman correlation (rank-based, more robust)
            spearman_corr, spearman_pvalue = stats.spearmanr(data[var1], data[var2])
            
            return {
                "variables": [var1, var2],
                "n_observations": len(data),
                "pearson": {
                    "correlation": round(float(corr), 4),
                    "p_value": round(float(pvalue), 6),
                    "significant": pvalue < 0.05,
                    "strength": self._interpret_correlation(corr)
                },
                "spearman": {
                    "correlation": round(float(spearman_corr), 4),
                    "p_value": round(float(spearman_pvalue), 6),
                    "significant": spearman_pvalue < 0.05
                }
            }
            
        except Exception as e:
            logger.error(f"Error in correlation test: {str(e)}")
            return {"error": str(e)}
    
    def simple_regression(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str
    ) -> Dict:
        """
        Perform simple linear regression
        
        Args:
            df: Input dataframe
            x_col: Independent variable
            y_col: Dependent variable
            
        Returns:
            Regression results
        """
        try:
            # Remove missing values
            data = df[[x_col, y_col]].dropna()
            
            if len(data) < 3:
                return {"error": "Need at least 3 observations"}
            
            x = data[x_col].values
            y = data[y_col].values
            
            # Perform regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Calculate predictions and residuals
            y_pred = slope * x + intercept
            residuals = y - y_pred
            
            # R-squared
            r_squared = r_value ** 2
            
            return {
                "model": "Simple Linear Regression",
                "independent_var": x_col,
                "dependent_var": y_col,
                "n_observations": len(data),
                "coefficients": {
                    "intercept": round(float(intercept), 4),
                    "slope": round(float(slope), 4),
                    "std_error": round(float(std_err), 4)
                },
                "statistics": {
                    "r_squared": round(float(r_squared), 4),
                    "correlation": round(float(r_value), 4),
                    "p_value": round(float(p_value), 6),
                    "significant": p_value < 0.05
                },
                "equation": f"{y_col} = {round(intercept, 2)} + {round(slope, 2)} * {x_col}",
                "interpretation": self._interpret_regression(r_squared, p_value)
            }
            
        except Exception as e:
            logger.error(f"Error in regression: {str(e)}")
            return {"error": str(e)}
    
    def _interpret_hypothesis_test(self, pvalue: float, alpha: float) -> str:
        """Interpret hypothesis test results"""
        if pvalue < alpha:
            return f"Reject null hypothesis (p < {alpha}). Groups have significantly different means."
        else:
            return f"Fail to reject null hypothesis (p >= {alpha}). No significant difference detected."
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def _interpret_regression(self, r_squared: float, pvalue: float) -> str:
        """Interpret regression results"""
        if pvalue >= 0.05:
            return "Relationship is not statistically significant."
        
        if r_squared < 0.3:
            strength = "weak"
        elif r_squared < 0.7:
            strength = "moderate"
        else:
            strength = "strong"
        
        return f"Statistically significant {strength} relationship (R² = {r_squared:.2f})"

