# app.py

import streamlit as st
import os
import tempfile
from typing import Optional, Dict, Any, List
from processors.document_processor import get_processor
from utils.vector_store import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from agents.excel_analysis_agent import ExcelAnalysisAgent
from utils.data_visualization import DataVisualizationManager
import plotly.express as px
import pandas as pd
import json
import plotly.graph_objects as go
from agents.prompts import EXCEL_ANALYSIS_PROMPT, VIZ_SYSTEM_PROMPT

def initialize_session_state():
    """Initialize session state variables"""
    session_vars = {
        'vector_store': None,
        'chat_history': [],
        'conversation': None,
        'excel_agent': None,
        'current_file': None,
        'uploaded_files': [],
        'analysis_history': [],
        'selected_files': set(),
        'excel_chat_history': [],
        'file_paths': {},
        'data_manager': None,  # For visualization manager object
        'dataframes': {}       # For storing dataframes
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def setup_chain(api_key: str):
    """Setup the conversation chain"""
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant skilled in data analysis and document understanding. "
            "Use the following context to answer questions and suggest insights when appropriate."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    
    memory = ConversationBufferMemory(return_messages=True)
    
    return ConversationChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

def process_file(file, api_key: str) -> Optional[Dict[str, Any]]:
    """Process uploaded file"""
    if not file:
        return None
        
    try:
        # Create a directory for uploaded files if it doesn't exist
        upload_dir = "uploaded_files"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getvalue())
        
        # Store the full path in session state
        if 'file_paths' not in st.session_state:
            st.session_state.file_paths = {}
        st.session_state.file_paths[file.name] = file_path
        
        # Process based on file type
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension in ['.xlsx', '.xls']:
            # Initialize data manager if not exists
            if not st.session_state.data_manager:
                st.session_state.data_manager = DataVisualizationManager()
            
            # Try to load the Excel file
            df = st.session_state.data_manager.get_dataframe(file.name)
            if df is None:
                st.error(f"Could not read Excel file: {file.name}")
                return None
            
            if file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(file.name)
            st.session_state.current_file = file.name
            
            return {
                'texts': [f"Excel file processed: {file.name}"],
                'metadata': [{'type': 'excel', 'filename': file.name}]
            }
        else:
            # Process other file types using the appropriate processor
            try:
                processor = get_processor(file_extension, api_key)
                if processor:
                    result = processor.process(file_path)
                    return result
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def render_excel_analysis_tab():
    st.title("Excel Data Analysis")
    
    # File upload section in a container
    with st.container():
        uploaded_files = st.file_uploader(
            "Upload Excel files to begin analysis",
            type=["xlsx", "xls"],
            accept_multiple_files=True
        )
    
    if uploaded_files:
        # Initialize data manager and agent if not exists
        if st.session_state.data_manager is None:
            st.session_state.data_manager = DataVisualizationManager()
        if st.session_state.excel_agent is None:
            st.session_state.excel_agent = ExcelAnalysisAgent(st.session_state.api_key)
        
        # Process uploaded files with progress bar
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            if file.name not in st.session_state.dataframes:
                try:
                    # Save file temporarily and store its path
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                        tmp_file.write(file.getvalue())
                        st.session_state.file_paths[file.name] = tmp_file.name
                    
                    # Read the dataframe
                    df = pd.read_excel(st.session_state.file_paths[file.name])
                    st.session_state.dataframes[file.name] = df
                    
                    # Update progress
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {str(e)}")
                    continue
        
        progress_bar.empty()  # Clear progress bar after completion
        
        # File selection
        selected_files = st.multiselect(
            "Select Files to Analyze",
            list(st.session_state.dataframes.keys())
        )
        
        if selected_files:
            # Create tabs for different analysis types
            tab1, tab2, tab3 = st.tabs(["Chat", "Visualization", "Smart Analysis"])
            
            # Chat Tab
            with tab1:
                st.subheader("Analysis Chat")
                # Initialize chat history if needed
                if 'excel_chat_history' not in st.session_state:
                    st.session_state.excel_chat_history = []
                
                # Chat input
                query = st.text_input("Ask about your data:")
                if query:
                    with st.spinner("Processing..."):
                        result = st.session_state.excel_agent.process_query(query, selected_files)
                        st.session_state.excel_chat_history.append(("You", query))
                        st.session_state.excel_chat_history.append(("Assistant", result))
                
                # Display chat history in a container with scrolling
                chat_container = st.container()
                with chat_container:
                    for role, msg in st.session_state.excel_chat_history:
                        if role == "You":
                            st.markdown(f"**You:** {msg}")
                        else:
                            if isinstance(msg, dict):
                                if msg['type'] == 'text':
                                    st.markdown(f"**Assistant:** {msg['content']}")
                                elif msg['type'] == 'error':
                                    st.error(msg['content'])
            
            # Visualization Tab
            with tab2:
                st.subheader("Create Visualizations")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    file_name = st.selectbox("Select file", selected_files)
                    if file_name:
                        df = st.session_state.dataframes[file_name]
                        plot_type = st.selectbox("Plot Type", ["bar", "line", "scatter", "pie", "histogram"])
                        cols = df.columns.tolist()
                        x_col = st.selectbox("X-axis", cols)
                        if plot_type != "histogram":
                            y_col = st.selectbox("Y-axis", cols)
                        else:
                            y_col = None
                
                with col2:
                    if st.button("Generate Plot"):
                        with st.spinner("Creating visualization..."):
                            try:
                                # Create figure directly using plotly
                                if plot_type == "bar":
                                    fig = px.bar(df, x=x_col, y=y_col)
                                elif plot_type == "line":
                                    fig = px.line(df, x=x_col, y=y_col)
                                elif plot_type == "scatter":
                                    fig = px.scatter(df, x=x_col, y=y_col)
                                elif plot_type == "pie":
                                    fig = px.pie(df, names=x_col, values=y_col)
                                else:  # histogram
                                    fig = px.histogram(df, x=x_col)
                                
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating visualization: {str(e)}")
            
            # Smart Analysis Tab
            with tab3:
                st.subheader("Smart Analysis")
                if st.button("Generate Complete Analysis"):
                    for file_name in selected_files:
                        with st.expander(f"Analysis for {file_name}", expanded=True):
                            df = st.session_state.dataframes[file_name]
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("Statistical Summary")
                                st.write(df.describe())
                            
                            with col2:
                                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                                if len(numeric_cols) > 1:
                                    st.write("Correlations")
                                    st.write(df[numeric_cols].corr())
                            
                            st.write("Key Visualizations")
                            viz_cols = st.columns(2)
                            for i, col in enumerate(numeric_cols):
                                with viz_cols[i % 2]:
                                    fig = st.session_state.data_manager.create_visualization(
                                        file_name, "histogram", col, None,
                                        f"Distribution of {col}"
                                    )
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)

def render_visualization_controls(selected_files: List[str]):
    """Render controls for visualization creation"""
    if not selected_files:
        st.warning("Please select at least one file to analyze")
        return
        
    if len(selected_files) == 1:
        # Single file visualization
        file_name = selected_files[0]
        
        # Initialize data manager if not exists
        if not st.session_state.data_manager:
            st.session_state.data_manager = DataVisualizationManager()
            
        # Get columns for the selected file
        columns = st.session_state.data_manager.get_available_columns(file_name)
        
        if not columns:
            st.error(f"Could not read columns from {file_name}. Please check if the file is valid.")
            return
            
        plot_type = st.selectbox(
            "Plot Type",
            ["line", "bar", "scatter", "box", "histogram", "pie"]
        )
        
        if plot_type != "pie":
            x_col = st.selectbox("X-axis", columns)
            y_col = st.selectbox("Y-axis", columns)
            
            if st.button("Generate Visualization"):
                fig = st.session_state.data_manager.create_visualization(
                    file_name, plot_type, x_col, y_col,
                    f"{plot_type.capitalize()} Plot of {y_col} vs {x_col}"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not create visualization. Please check your data.")
        else:
            value_col = st.selectbox("Values", columns)
            name_col = st.selectbox("Names", columns)
            
            if st.button("Generate Pie Chart"):
                fig = st.session_state.data_manager.create_visualization(
                    file_name, "pie", name_col, value_col,
                    f"Pie Chart of {value_col} by {name_col}"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Could not create pie chart. Please check your data.")
    
    else:
        # Multi-file visualization
        st.write("Comparing multiple files:")
        plot_type = st.selectbox(
            "Plot Type",
            ["line", "bar", "scatter", "box"]
        )
        
        columns = {}
        for file in selected_files:
            cols = DataVisualizationManager.get_available_columns(file)
            st.write(f"Columns for {file}:")
            columns[file] = {
                'x': st.selectbox(f"X-axis for {file}", cols, key=f"x_{file}"),
                'y': st.selectbox(f"Y-axis for {file}", cols, key=f"y_{file}")
            }
        
        if st.button("Generate Comparison"):
            fig = DataVisualizationManager.create_comparison_visualization(
                selected_files, columns, plot_type,
                "Comparison Across Files"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

def render_statistics_controls(selected_files: List[str]):
    """Render controls for statistical analysis"""
    if st.button("Generate Statistical Summary"):
        for file in selected_files:
            stats = DataVisualizationManager.get_summary_stats(file)
            with st.expander(f"Statistics for {file}"):
                st.json(stats)

def render_comparison_controls(selected_files: List[str]):
    """Render controls for data comparison"""
    if len(selected_files) < 2:
        st.warning("Select at least two files for comparison")
        return
    
    if st.button("Compare Datasets"):
        results = {}
        for file in selected_files:
            df = DataVisualizationManager.get_dataframe(file)
            if df is not None:
                results[file] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'numeric_columns': df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                }
        st.json(results)

def process_excel_query(query: str, selected_files: List[str]):
    """Process a natural language query about Excel data"""
    if not st.session_state.excel_agent:
        st.error("Excel analysis agent not initialized")
        return
    
    result = st.session_state.excel_agent.process_query(query, selected_files)
    
    if result:
        st.session_state.analysis_history.append({
            'query': query,
            'result': result,
            'timestamp': pd.Timestamp.now()
        })

def render_analysis_result(result: Dict[str, Any]):
    """Render an analysis result"""
    if result['type'] == 'text':
        st.write(result['content'])
    elif result['type'] == 'visualization':
        st.plotly_chart(result['content'], use_container_width=True)
    elif result['type'] == 'multiple_visualizations':
        for viz in result['content']:
            st.plotly_chart(viz['content'], use_container_width=True)
            if viz.get('message'):
                st.info(viz['message'])
    elif result['type'] == 'error':
        st.error(result['content'])
    
    if result.get('message'):
        st.info(result['message'])

def cleanup_selected_files():
    """Remove any selected files that no longer exist in uploaded_files"""
    if 'selected_files' in st.session_state:
        excel_files = set(f for f in st.session_state.uploaded_files if f.endswith(('.xlsx', '.xls')))
        st.session_state.selected_files = st.session_state.selected_files.intersection(excel_files)

def main():
    st.set_page_config(page_title="Document Chat & Data Analysis", layout="wide")
    
    initialize_session_state()
    
    # Sidebar for API key
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            if not st.session_state.vector_store:
                st.session_state.vector_store = VectorStore(api_key)
            if not st.session_state.conversation:
                st.session_state.conversation = setup_chain(api_key)
            if not st.session_state.excel_agent:
                st.session_state.excel_agent = ExcelAnalysisAgent(api_key)
            if not st.session_state.data_manager:
                st.session_state.data_manager = DataVisualizationManager()
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["💬 Document Chat", "📊 Excel Analysis"])
    
    with tab1:
        st.title("Document Chat Assistant")
        
        # File upload section
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=["pdf", "xlsx", "xls", "mp3", "wav", "png", "jpg", "jpeg", "docx", "doc"]
            )
            
            if uploaded_file:
                with st.spinner("Processing document..."):
                    result = process_file(uploaded_file, api_key)
                    if result:
                        st.session_state.vector_store.add_texts(
                            result['texts'],
                            result['metadata']
                        )
                        st.success("Document processed successfully!")
        
        with col2:
            url = st.text_input("Or enter a URL to process:")
            if url and url.strip():
                with st.spinner("Processing webpage..."):
                    processor = get_processor('web', api_key)
                    result = processor.process(url)
                    if result:
                        if result['metadata'][0].get('type') == 'error':
                            st.error(result['texts'][0])
                        else:
                            st.session_state.vector_store.add_texts(
                                result['texts'],
                                result['metadata']
                            )
                            st.success("Webpage processed successfully!")
        
        # Chat interface
        st.markdown("---")
        if st.session_state.vector_store and hasattr(st.session_state.vector_store, 'vector_store'):
            query = st.text_input("Ask a question about your documents:")
            if query:
                with st.spinner("Processing..."):
                    docs = st.session_state.vector_store.similarity_search(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    response = st.session_state.conversation.predict(
                        input=f"Context: {context}\n\nQuestion: {query}"
                    )
                    
                    st.session_state.chat_history.append(("You", query))
                    st.session_state.chat_history.append(("Assistant", response))
        
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("### Chat History")
                for role, message in st.session_state.chat_history:
                    if role == "You":
                        st.markdown(f"**You:** {message}")
                    else:
                        st.markdown(f"**Assistant:** {message}")
    
    with tab2:
        render_excel_analysis_tab()

class ExcelAnalysisAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analysis of the dataframe"""
        analysis = {
            'basic_stats': {},
            'correlations': None,
            'suggested_visualizations': []
        }
        
        # Basic statistics
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        analysis['basic_stats'] = {
            'numeric': {col: {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            } for col in numeric_cols},
            'categorical': {col: df[col].value_counts().to_dict() for col in categorical_cols}
        }
        
        # Correlations for numeric columns
        if len(numeric_cols) > 1:
            analysis['correlations'] = df[numeric_cols].corr().to_dict()
        
        # Suggest visualizations based on data types
        for col in numeric_cols:
            analysis['suggested_visualizations'].append({
                'type': 'histogram',
                'columns': [col],
                'title': f'Distribution of {col}'
            })
        
        for col in categorical_cols:
            analysis['suggested_visualizations'].append({
                'type': 'pie',
                'columns': [col],
                'title': f'Distribution of {col}'
            })
            
            # Bar charts for categorical vs numeric
            for num_col in numeric_cols:
                analysis['suggested_visualizations'].append({
                    'type': 'bar',
                    'columns': [col, num_col],
                    'title': f'{num_col} by {col}'
                })
        
        return analysis
    
    def process_query(self, query: str, selected_files: List[str]) -> Dict[str, Any]:
        try:
            if "complete" in query.lower() and "analysis" in query.lower():
                results = []
                for file in selected_files:
                    df = st.session_state.data_manager.get_dataframe(file)
                    if df is not None:
                        analysis = self.analyze_data(df)
                        
                        # Create visualizations based on analysis
                        visualizations = []
                        for viz_spec in analysis['suggested_visualizations']:
                            fig = st.session_state.data_manager.create_visualization(
                                file,
                                viz_spec['type'],
                                viz_spec['columns'][0],
                                viz_spec['columns'][1] if len(viz_spec['columns']) > 1 else None,
                                viz_spec['title']
                            )
                            if fig:
                                visualizations.append({
                                    'type': 'visualization',
                                    'content': fig,
                                    'message': viz_spec['title']
                                })
                        
                        results.append({
                            'type': 'analysis',
                            'content': {
                                'statistics': analysis['basic_stats'],
                                'correlations': analysis['correlations'],
                                'visualizations': visualizations
                            },
                            'message': f"Complete analysis for {file}"
                        })
                
                return {
                    'type': 'multiple_analyses',
                    'content': results
                }
            
            # Original query processing logic
            return super().process_query(query, selected_files)
            
        except Exception as e:
            return {
                'type': 'error',
                'content': f"Error processing query: {str(e)}"
            }

if __name__ == "__main__":
    main()