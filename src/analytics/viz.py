"""
Data Visualization
Showcases: Data Analysis with Spreadsheets and SQL (Meta)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Tuple
import io
import base64
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Create data visualizations"""
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.fig_size = (10, 6)
    
    def plot_distribution(
        self, 
        df: pd.DataFrame, 
        column: str,
        bins: int = 30
    ) -> go.Figure:
        """
        Create distribution plot
        
        Args:
            df: Input dataframe
            column: Column to plot
            bins: Number of bins for histogram
            
        Returns:
            Plotly figure
        """
        try:
            fig = px.histogram(
                df, 
                x=column,
                nbins=bins,
                title=f"Distribution of {column}",
                labels={column: column, "count": "Frequency"}
            )
            
            # Add mean and median lines
            mean_val = df[column].mean()
            median_val = df[column].median()
            
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_val:.2f}"
            )
            fig.add_vline(
                x=median_val, 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"Median: {median_val:.2f}"
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plot: {str(e)}")
            return None
    
    def plot_correlation_heatmap(
        self, 
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            df: Input dataframe
            columns: Specific columns to include (optional)
            
        Returns:
            Plotly figure
        """
        try:
            # Select numeric columns
            if columns:
                numeric_df = df[columns]
            else:
                numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                logger.warning("No numeric columns for correlation")
                return None
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            
            # Add correlation values as text
            fig.update_traces(
                text=corr_matrix.round(2).values,
                texttemplate='%{text}'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def plot_scatter(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str,
        color_col: Optional[str] = None,
        trendline: bool = True
    ) -> go.Figure:
        """
        Create scatter plot
        
        Args:
            df: Input dataframe
            x_col: X-axis column
            y_col: Y-axis column
            color_col: Column for color coding (optional)
            trendline: Add trendline (default: True)
            
        Returns:
            Plotly figure
        """
        try:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col}",
                trendline="ols" if trendline else None,
                labels={x_col: x_col, y_col: y_col}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return None
    
    def plot_bar_chart(
        self, 
        df: pd.DataFrame, 
        x_col: str, 
        y_col: str,
        top_n: int = 10
    ) -> go.Figure:
        """
        Create bar chart
        
        Args:
            df: Input dataframe
            x_col: X-axis column (categories)
            y_col: Y-axis column (values)
            top_n: Show top N categories
            
        Returns:
            Plotly figure
        """
        try:
            # Aggregate and sort
            agg_df = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(top_n)
            
            fig = px.bar(
                x=agg_df.index,
                y=agg_df.values,
                title=f"Top {top_n} {x_col} by {y_col}",
                labels={"x": x_col, "y": y_col}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return None
    
    def plot_time_series(
        self, 
        df: pd.DataFrame, 
        date_col: str, 
        value_col: str,
        resample_freq: Optional[str] = None
    ) -> go.Figure:
        """
        Create time series plot
        
        Args:
            df: Input dataframe
            date_col: Date column
            value_col: Value column
            resample_freq: Resampling frequency (e.g., 'D', 'W', 'M')
            
        Returns:
            Plotly figure
        """
        try:
            # Ensure date column is datetime
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            # Resample if specified
            if resample_freq:
                df_copy = df_copy.set_index(date_col).resample(resample_freq)[value_col].mean().reset_index()
            
            fig = px.line(
                df_copy,
                x=date_col,
                y=value_col,
                title=f"{value_col} Over Time",
                labels={date_col: "Date", value_col: value_col}
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            return None
    
    def plot_box_plot(
        self, 
        df: pd.DataFrame, 
        column: str,
        group_by: Optional[str] = None
    ) -> go.Figure:
        """
        Create box plot
        
        Args:
            df: Input dataframe
            column: Column to plot
            group_by: Group by column (optional)
            
        Returns:
            Plotly figure
        """
        try:
            if group_by:
                fig = px.box(
                    df,
                    x=group_by,
                    y=column,
                    title=f"{column} Distribution by {group_by}",
                    labels={group_by: group_by, column: column}
                )
            else:
                fig = px.box(
                    df,
                    y=column,
                    title=f"{column} Distribution",
                    labels={column: column}
                )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating box plot: {str(e)}")
            return None
    
    def create_summary_dashboard(self, df: pd.DataFrame) -> List[go.Figure]:
        """
        Create a summary dashboard with multiple plots
        
        Args:
            df: Input dataframe
            
        Returns:
            List of Plotly figures
        """
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Correlation heatmap
            corr_fig = self.plot_correlation_heatmap(df)
            if corr_fig:
                figures.append(corr_fig)
            
            # Distribution of first numeric column
            dist_fig = self.plot_distribution(df, numeric_cols[0])
            if dist_fig:
                figures.append(dist_fig)
            
            # Box plot of first numeric column
            box_fig = self.plot_box_plot(df, numeric_cols[0])
            if box_fig:
                figures.append(box_fig)
        
        return figures

