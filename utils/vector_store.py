from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """Add texts to the vector store"""
        # Split texts into chunks
        all_chunks = []
        all_metadatas = []

        for idx, text in enumerate(texts):
            chunks = self.text_splitter.split_text(text)
            all_chunks.extend(chunks)
            
            if metadatas:
                chunk_metadata = [metadatas[idx]] * len(chunks)
                all_metadatas.extend(chunk_metadata)

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                metadatas=all_metadatas if metadatas else None
            )
        else:
            self.vector_store.add_texts(
                texts=all_chunks,
                metadatas=all_metadatas if metadatas else None
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Please add documents first.")
            
        return self.vector_store.similarity_search(query, k=k)

    def clear(self) -> None:
        """Clear the vector store"""
        self.vector_store = None