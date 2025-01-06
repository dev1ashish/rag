# src/agents/excel_analysis/agent.py

from typing import List, Dict, Any, Optional
import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils.data_visualization import DataVisualizationManager
from .prompts import EXCEL_ANALYSIS_PROMPT, VIZ_SYSTEM_PROMPT

class ExcelAnalysisAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo"
        )
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=EXCEL_ANALYSIS_PROMPT
        )
        self.visualization_manager = DataVisualizationManager()

    def process_query(self, query: str, selected_files: List[str]) -> Dict[str, Any]:
        try:
            # Get data context for all selected files
            context = self._get_data_context(selected_files)
            
            # Format the prompt with context and query
            formatted_query = f"""
            Based on the following data:
            {context}
            
            Please answer this question: {query}
            
            If the answer requires visualization, please indicate so in your response.
            """
            
            # Get response from conversation chain
            response = self.conversation.predict(input=formatted_query)
            
            return {
                'type': 'text',
                'content': response
            }
            
        except Exception as e:
            return {
                'type': 'error',
                'content': f"Error processing query: {str(e)}"
            }

    def _get_data_context(self, selected_files: List[str]) -> str:
        context_parts = []
        for file in selected_files:
            df = st.session_state.dataframes[file]
            context_parts.extend([
                f"\nFile: {file}",
                f"Number of rows: {len(df)}",
                f"Columns: {', '.join(df.columns.tolist())}",
                "Sample data:",
                df.head().to_string(),
                "Summary statistics:",
                df.describe().to_string(),
                "---"
            ])
        return "\n".join(context_parts)

    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get the history of analyses performed"""
        if not hasattr(st.session_state, 'analysis_history'):
            st.session_state.analysis_history = []
        return st.session_state.analysis_history

    def add_to_history(self, query: str, result: Dict[str, Any]):
        """Add an analysis result to the history"""
        if not hasattr(st.session_state, 'analysis_history'):
            st.session_state.analysis_history = []
            
        st.session_state.analysis_history.append({
            'query': query,
            'result': result,
            'timestamp': pd.Timestamp.now()
        })