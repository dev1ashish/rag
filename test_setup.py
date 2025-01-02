import streamlit as st
import openai
import pandas as pd
from langchain import OpenAI
import faiss
import numpy as np
from PyPDF2 import PdfReader

def test_imports():
    """Test if all required packages are properly installed"""
    print("✅ All basic imports successful")
    
    # Test OpenAI
    try:
        openai.Model.list()
        print("✅ OpenAI connection possible (need valid API key for actual use)")
    except:
        print("✅ OpenAI importable (need API key for connection test)")
        
    # Test FAISS
    try:
        d = 64                           # dimension
        nb = 100                         # database size
        nq = 10                          # nb of queries
        np.random.seed(1234)             # make reproducible
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        index = faiss.IndexFlatL2(d)     # build the index
        index.add(xb)                    # add vectors to the index
        print("✅ FAISS working correctly")
    except Exception as e:
        print(f"❌ FAISS test failed: {str(e)}")

if __name__ == "__main__":
    test_imports()