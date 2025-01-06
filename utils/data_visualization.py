# src/utils/data_visualization.py

from typing import Dict, Any, Optional, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
import numpy as np

class DataVisualizationManager:
    """Manages data visualization with enhanced Excel support"""
    
    def __init__(self):
        self.dataframes = {}
    
    def get_dataframe(self, file_name: str) -> Optional[pd.DataFrame]:
        """Get dataframe from file name"""
        if file_name not in self.dataframes:
            try:
                # Get the full path from session state
                file_path = st.session_state.file_paths.get(file_name)
                if not file_path:
                    print(f"File path not found for {file_name}")
                    return None
                
                # Read the Excel file from the correct path
                self.dataframes[file_name] = pd.read_excel(file_path)
                print(f"Successfully read {file_name} from {file_path}")
                
            except Exception as e:
                print(f"Error reading file {file_name}: {str(e)}")
                return None
        return self.dataframes[file_name]
    
    def create_visualization(self, file_name: str, plot_type: str, x_col: str, 
                           y_col: Optional[str] = None, title: str = "") -> Optional[go.Figure]:
        """Create visualization from dataframe"""
        df = self.get_dataframe(file_name)
        if df is None:
            return None
            
        try:
            if plot_type == "pie":
                fig = px.pie(df, values=y_col, names=x_col, title=title)
            elif plot_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=title)
            elif plot_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=title)
            elif plot_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=title)
            elif plot_type == "histogram":
                fig = px.histogram(df, x=x_col, title=title)
            else:
                fig = None
                
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

    @staticmethod
    def store_dataframe(df: pd.DataFrame, file_name: str) -> None:
        """Store DataFrame in session state with enhanced metadata"""
        if 'dataframes' not in st.session_state:
            st.session_state.dataframes = {}
            st.session_state.dataframe_metadata = {}
            st.session_state.dataframe_relationships = {}
        
        # Store the DataFrame
        st.session_state.dataframes[file_name] = df
        
        # Store metadata about structure
        st.session_state.dataframe_metadata[file_name] = {
            'columns': df.columns.tolist(),
            'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape,
            'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'summary_stats': DataVisualizationManager._generate_summary_stats(df)
        }
        
        # Update relationships with other DataFrames
        DataVisualizationManager.update_relationships(file_name, df)

    @staticmethod
    def _generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for numeric columns"""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing': int(df[col].isna().sum())
            }
        return stats

    @staticmethod
    def create_comparison_visualization(files: List[str], columns: Dict[str, str], 
                                     plot_type: str, title: str = "") -> Optional[go.Figure]:
        """Create visualization comparing data from multiple files"""
        try:
            if plot_type in ["line", "scatter", "bar"]:
                fig = go.Figure()
                
                for file_name in files:
                    df = DataVisualizationManager.get_dataframe(file_name)
                    if df is not None:
                        x_col = columns[file_name]['x']
                        y_col = columns[file_name]['y']
                        
                        if plot_type == "line":
                            fig.add_trace(
                                go.Scatter(x=df[x_col], y=df[y_col],
                                         mode='lines+markers',
                                         name=f"{file_name}: {y_col}")
                            )
                        elif plot_type == "scatter":
                            fig.add_trace(
                                go.Scatter(x=df[x_col], y=df[y_col],
                                         mode='markers',
                                         name=f"{file_name}: {y_col}")
                            )
                        elif plot_type == "bar":
                            fig.add_trace(
                                go.Bar(x=df[x_col], y=df[y_col],
                                      name=f"{file_name}: {y_col}")
                            )
                
                fig.update_layout(
                    title=title,
                    height=600,
                    showlegend=True,
                    template="plotly_white"
                )
                
                return fig
            
            elif plot_type == "box":
                fig = make_subplots(rows=1, cols=len(files),
                                  subplot_titles=[f"Distribution in {f}" for f in files])
                
                for idx, file_name in enumerate(files, 1):
                    df = DataVisualizationManager.get_dataframe(file_name)
                    if df is not None:
                        y_col = columns[file_name]['y']
                        fig.add_trace(
                            go.Box(y=df[y_col],
                                  name=f"{file_name}: {y_col}"),
                            row=1, col=idx
                        )
                
                fig.update_layout(
                    title=title,
                    height=500,
                    showlegend=True,
                    template="plotly_white"
                )
                
                return fig
                
        except Exception as e:
            print(f"Error creating comparison visualization: {str(e)}")
            return None

    @staticmethod
    def create_visualization_from_query(query: str, file_name: str) -> Optional[go.Figure]:
        """Create visualization based on natural language query"""
        df = DataVisualizationManager.get_dataframe(file_name)
        if df is None:
            return None
            
        try:
            # Determine plot type from query
            plot_type = "line"  # default
            if any(word in query.lower() for word in ["bar", "column"]):
                plot_type = "bar"
            elif any(word in query.lower() for word in ["scatter", "correlation"]):
                plot_type = "scatter"
            elif any(word in query.lower() for word in ["box", "distribution"]):
                plot_type = "box"
            elif any(word in query.lower() for word in ["histogram", "distribution"]):
                plot_type = "histogram"
            
            # Get columns mentioned in query
            columns = DataVisualizationManager.get_available_columns(file_name)
            mentioned_columns = [col for col in columns if col.lower() in query.lower()]
            
            if len(mentioned_columns) >= 2:
                # For two-variable plots
                return DataVisualizationManager.create_visualization(
                    file_name,
                    plot_type,
                    mentioned_columns[0],
                    mentioned_columns[1],
                    f"{plot_type.capitalize()} Plot based on query"
                )
            elif len(mentioned_columns) == 1 and plot_type in ["histogram", "box"]:
                # For single-variable plots
                return DataVisualizationManager.create_visualization(
                    file_name,
                    plot_type,
                    mentioned_columns[0],
                    mentioned_columns[0],
                    f"{plot_type.capitalize()} Plot of {mentioned_columns[0]}"
                )
                
            return None
            
        except Exception as e:
            print(f"Error creating visualization from query: {str(e)}")
            return None

    # Continuing src/utils/data_visualization.py

    def get_available_columns(self, file_name: str) -> List[str]:
        """Get list of available columns in the dataframe"""
        df = self.get_dataframe(file_name)
        if df is not None:
            return df.columns.tolist()
        return []

    @staticmethod
    def get_summary_stats(file_name: str) -> Dict[str, Any]:
        """Get summary statistics for a file"""
        if 'dataframe_metadata' in st.session_state:
            metadata = st.session_state.dataframe_metadata.get(file_name, {})
            return metadata.get('summary_stats', {})
        return {}

    @staticmethod
    def analyze_distributions(file_name: str) -> Optional[go.Figure]:
        """Create distribution plots for numeric columns"""
        df = DataVisualizationManager.get_dataframe(file_name)
        if df is None:
            return None

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) == 0:
            return None

        try:
            # Create subplots for each numeric column
            fig = make_subplots(
                rows=len(numeric_cols), 
                cols=1,
                subplot_titles=[f"Distribution of {col}" for col in numeric_cols]
            )

            for idx, col in enumerate(numeric_cols, 1):
                fig.add_trace(
                    go.Histogram(x=df[col], name=col),
                    row=idx, col=1
                )

            fig.update_layout(
                height=300 * len(numeric_cols),
                showlegend=False,
                title_text=f"Distributions in {file_name}",
                template="plotly_white"
            )

            return fig
        except Exception as e:
            print(f"Error analyzing distributions: {str(e)}")
            return None

    @staticmethod
    def create_correlation_matrix(file_name: str) -> Optional[go.Figure]:
        """Create correlation matrix heatmap"""
        df = DataVisualizationManager.get_dataframe(file_name)
        if df is None:
            return None

        try:
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.empty:
                return None

            corr_matrix = numeric_df.corr()

            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                aspect="auto",
                title=f"Correlation Matrix - {file_name}"
            )

            fig.update_layout(
                height=600,
                width=800,
                template="plotly_white"
            )

            return fig
        except Exception as e:
            print(f"Error creating correlation matrix: {str(e)}")
            return None

    @staticmethod
    def save_visualization(fig: go.Figure, filename: str) -> bool:
        """Save visualization to file"""
        try:
            if not filename.endswith(('.png', '.html')):
                filename += '.html'
            
            if filename.endswith('.png'):
                fig.write_image(filename)
            else:
                fig.write_html(filename)
            return True
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
            return False

    @staticmethod
    def suggest_visualizations(file_name: str) -> List[Dict[str, Any]]:
        """Generate intelligent visualization suggestions"""
        df = DataVisualizationManager.get_dataframe(file_name)
        if df is None:
            return []

        suggestions = []
        metadata = st.session_state.dataframe_metadata.get(file_name, {})
        numeric_cols = metadata.get('numeric_columns', [])
        categorical_cols = metadata.get('categorical_columns', [])

        # Numeric column relationships
        if len(numeric_cols) >= 2:
            suggestions.append({
                'type': 'scatter',
                'columns': numeric_cols[:2],
                'message': f"Consider a scatter plot to explore relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            })

        # Categorical vs Numeric
        if categorical_cols and numeric_cols:
            suggestions.append({
                'type': 'box',
                'columns': [categorical_cols[0], numeric_cols[0]],
                'message': f"Box plot showing distribution of {numeric_cols[0]} across {categorical_cols[0]} categories"
            })

        # Time series if date column exists
        date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
        if date_cols and numeric_cols:
            suggestions.append({
                'type': 'line',
                'columns': [date_cols[0], numeric_cols[0]],
                'message': f"Time series plot showing {numeric_cols[0]} over time"
            })

        # Distribution analysis
        if numeric_cols:
            suggestions.append({
                'type': 'histogram',
                'columns': [numeric_cols[0]],
                'message': f"Analyze distribution of {numeric_cols[0]}"
            })

        return suggestions

    @staticmethod
    def generate_insights(file_name: str) -> List[str]:
        """Generate basic insights about the data"""
        df = DataVisualizationManager.get_dataframe(file_name)
        if df is None:
            return []

        insights = []
        metadata = st.session_state.dataframe_metadata.get(file_name, {})
        stats = metadata.get('summary_stats', {})

        # Basic data insights
        insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

        # Missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            insights.append(f"Found missing values in columns: {', '.join(missing_cols)}")

        # Numeric column insights
        for col, col_stats in stats.items():
            insights.append(f"{col}:")
            insights.append(f"  - Range: {col_stats['min']:.2f} to {col_stats['max']:.2f}")
            insights.append(f"  - Average: {col_stats['mean']:.2f}")
            if col_stats['missing'] > 0:
                insights.append(f"  - Missing values: {col_stats['missing']}")

        return insights
    
    @staticmethod
    def update_relationships(new_file: str, new_df: pd.DataFrame) -> None:
        """Update relationship metadata between DataFrames"""
        try:
            if 'dataframe_relationships' not in st.session_state:
                st.session_state.dataframe_relationships = {}
            
            relationships = st.session_state.dataframe_relationships
            
            for existing_file, existing_df in st.session_state.dataframes.items():
                if existing_file != new_file:
                    common_columns = set(new_df.columns) & set(existing_df.columns)
                    if common_columns:
                        if new_file not in relationships:
                            relationships[new_file] = {}
                        relationships[new_file][existing_file] = {
                            'common_columns': list(common_columns),
                            'potential_joins': DataVisualizationManager._analyze_join_possibilities(
                                new_df, existing_df, common_columns
                            )
                        }
        except Exception as e:
            print(f"Error updating relationships: {str(e)}")
            # Don't raise the error, just log it to avoid breaking the app flow