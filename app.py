import streamlit as st
import os
import tempfile
from typing import Optional, Dict, Any
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

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None

def setup_chain(api_key: str):
    """Setup the conversation chain"""
    llm = ChatOpenAI(
        temperature=0.7,
        openai_api_key=api_key,
        model_name="gpt-3.5-turbo"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Use the following context to answer questions."
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
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Creating temporary file...")
            progress_bar.progress(20)
            
            # Save uploaded file
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, 'wb') as f:
                f.write(file.getvalue())
            
            status_text.text("Initializing processor...")
            progress_bar.progress(40)
            
            # Get file extension for processor type
            file_extension = os.path.splitext(file.name)[1].lower()
            
            # Get appropriate processor
            try:
                processor = get_processor(file_extension, api_key)
            except ValueError as ve:
                st.error(str(ve))
                return None
            
            if not processor:
                st.error("Could not initialize processor")
                return None
            
            status_text.text(f"Processing document...")
            progress_bar.progress(60)
            
            try:
                result = processor.process(temp_file_path)
                if result['metadata'][0].get('type') == 'error':
                    st.error(result['metadata'][0].get('error', 'Unknown error processing file'))
                    return None
                
                status_text.text("Finalizing...")
                progress_bar.progress(100)
                
                return result
                
            except ValueError as ve:
                st.error(str(ve))
                return None
            finally:
                # Clean up
                status_text.empty()
                progress_bar.empty()
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Document Chat", layout="wide")
    st.title("Document Chat Assistant")
    
    initialize_session_state()
    
    # Sidebar for API key
    with st.sidebar:
        api_key = st.text_input("OpenAI API Key:", type="password")
        if api_key:
            if not st.session_state.vector_store:
                st.session_state.vector_store = VectorStore(api_key)
            if not st.session_state.conversation:
                st.session_state.conversation = setup_chain(api_key)
    
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return
    
    # Two-column layout for input
    col1, col2 = st.columns(2)
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, Excel, Office, Audio, or Image file)",
            type=["pdf", "xlsx", "xls", "mp3", "wav", "png", "jpg", "jpeg",
                "docx", "doc"]
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
        # URL input
        url = st.text_input("Or enter a URL to process:")
        if url and url.strip():
            with st.spinner("Processing webpage..."):
                processor = get_processor('web', api_key)
                result = processor.process(url)
                if result:
                    # Check for error type in metadata
                    if result['metadata'][0].get('type') == 'error':
                        if result['metadata'][0].get('error') == 'forbidden_access':
                            st.error(result['texts'][0])  # Display the forbidden access message
                        else:
                            st.error(f"Error: {result['metadata'][0].get('message', 'Unknown error')}")
                    else:
                        st.session_state.vector_store.add_texts(
                            result['texts'],
                            result['metadata']
                        )
                        st.success("Webpage processed successfully!")
    
    # Chat interface - moved outside columns
    st.markdown("---")  # Add a separator
    if st.session_state.vector_store and hasattr(st.session_state.vector_store, 'vector_store'):
        query = st.text_input("Ask a question about your documents:")
        if query:
            with st.spinner("Thinking..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get relevant documents
                status_text.text("Searching relevant documents...")
                progress_bar.progress(30)
                docs = st.session_state.vector_store.similarity_search(query)
                
                # Create context from documents
                status_text.text("Analyzing context...")
                progress_bar.progress(60)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Get response from conversation chain
                status_text.text("Generating response...")
                progress_bar.progress(90)
                response = st.session_state.conversation.predict(
                    input=f"Context: {context}\n\nQuestion: {query}"
                )
                
                # Clean up progress indicators
                status_text.empty()
                progress_bar.progress(100)
                progress_bar.empty()
                
                # Add to chat history
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

if __name__ == "__main__":
    main()