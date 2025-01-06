# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# from pathlib import Path
# import tempfile
# from processors.document_processor import DocumentProcessor
# from processors.office_processor import OfficeProcessor
# from utils.data_visualization import DataVisualizationManager
# from utils.vector_store import VectorStore

# def test_imports():
#     """Test if all required packages are properly installed"""
#     try:
#         import streamlit
#         import openai
#         import pandas
#         import langchain
#         import faiss
#         import numpy
#         from PyPDF2 import PdfReader
#         from docx import Document
#         from pptx import Presentation
#         print("✅ All basic imports successful")
#     except Exception as e:
#         print(f"❌ Import failed: {str(e)}")

# def test_office_processor():
#     """Test Office document processing"""
#     processor = OfficeProcessor()
    
#     # Create a test Word document
#     doc = Document()
#     doc.add_paragraph("Test paragraph 1")
#     doc.add_paragraph("Test paragraph 2")
    
#     # Add a table
#     table = doc.add_table(rows=2, cols=2)
#     table.cell(0, 0).text = "Cell 1"
#     table.cell(0, 1).text = "Cell 2"
#     table.cell(1, 0).text = "Cell 3"
#     table.cell(1, 1).text = "Cell 4"
    
#     # Save temporary file
#     with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
#         doc.save(tmp.name)
#         try:
#             result = processor.process(tmp.name)
#             if result and 'texts' in result and 'metadata' in result:
#                 print("✅ Office processor (DOCX) working correctly")
#             else:
#                 print("❌ Office processor (DOCX) failed")
#         except Exception as e:
#             print(f"❌ Office processor test failed: {str(e)}")
#         finally:
#             os.unlink(tmp.name)

# def test_data_visualization():
#     """Test data visualization functionality"""
#     try:
#         # Create test DataFrame
#         df = pd.DataFrame({
#             'x': np.random.rand(10),
#             'y': np.random.rand(10)
#         })
        
#         # Initialize session state
#         if 'dataframe_metadata' not in st.session_state:
#             st.session_state.dataframe_metadata = {}
        
#         # Store test data
#         test_file = "test_data.xlsx"
#         st.session_state.dataframe_metadata[test_file] = {
#             'columns': df.columns.tolist(),
#             'summary_stats': df.describe().to_dict()
#         }
        
#         # Test getting columns
#         columns = DataVisualizationManager.get_available_columns(test_file)
#         if columns == ['x', 'y']:
#             print("✅ Data visualization manager working correctly")
#         else:
#             print("❌ Data visualization manager failed")
            
#     except Exception as e:
#         print(f"❌ Data visualization test failed: {str(e)}")

# def test_vector_store():
#     """Test vector store functionality"""
#     try:
#         # Use a dummy API key for testing
#         vector_store = VectorStore("dummy_api_key")
        
#         # Test basic initialization
#         if hasattr(vector_store, 'embeddings') and hasattr(vector_store, 'index'):
#             print("✅ Vector store initialization successful")
#         else:
#             print("❌ Vector store initialization failed")
#     except Exception as e:
#         print(f"❌ Vector store test failed: {str(e)}")

# def main():
#     """Run all tests"""
#     print("Starting tests...\n")
    
#     print("1. Testing imports...")
#     test_imports()
#     print()
    
#     print("2. Testing Office Processor...")
#     test_office_processor()
#     print()
    
#     print("3. Testing Data Visualization...")
#     test_data_visualization()
#     print()
    
#     print("4. Testing Vector Store...")
#     test_vector_store()
#     print()
    
#     print("Tests completed!")

# if __name__ == "__main__":
#     main() 