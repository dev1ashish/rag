from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import List, Dict, Any, Optional
import copy

class VectorStore:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        
        # Custom text splitters for different document types by god whyy
        self.splitters = {
            'default': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            ),
            'presentation': RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                separators=["[Slide", "\n\n", "\n", " ", ""]
            ),
            'document': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            ),
            'office': RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        }

    def _get_splitter(self, doc_type: str) -> RecursiveCharacterTextSplitter:
        """Get appropriate text splitter based on document type"""
        return self.splitters.get(doc_type, self.splitters['default'])

    def _enhance_metadata(self, metadata: Dict[str, Any], chunk_idx: int, total_chunks: int) -> Dict[str, Any]:
        """Enhance metadata for a specific chunk"""
        enhanced = copy.deepcopy(metadata)
        enhanced['chunk'] = {
            'index': chunk_idx,
            'total': total_chunks
        }
        return enhanced

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the vector store with enhanced processing"""
        if not texts:
            return

        all_chunks = []
        all_metadatas = []

        for idx, text in enumerate(texts):
            # Get document type and sahi wala splitter
            doc_type = 'default'
            if metadatas and idx < len(metadatas):
                doc_type = metadatas[idx].get('type', 'default')
            
            splitter = self._get_splitter(doc_type)
            chunks = splitter.split_text(text)

            # Handle metadata for chonks
            if metadatas and idx < len(metadatas):
                base_metadata = metadatas[idx]
                # Create metadata for each chonk
                for chunk_idx, _ in enumerate(chunks):
                    chunk_metadata = self._enhance_metadata(
                        base_metadata,
                        chunk_idx,
                        len(chunks)
                    )
                    all_metadatas.append(chunk_metadata)
            else:
                # If no metadata, create basic metadata for each chunk, else crazy indexing error
                for chunk_idx, _ in enumerate(chunks):
                    all_metadatas.append({
                        'chunk': {'index': chunk_idx, 'total': len(chunks)},
                        'type': 'default'
                    })

            all_chunks.extend(chunks)

        # Create or update vector store gawd
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=all_chunks,
                embedding=self.embeddings,
                metadatas=all_metadatas
            )
        else:
            self.vector_store.add_texts(
                texts=all_chunks,
                metadatas=all_metadatas
            )

    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.vector_store is None:
            raise ValueError("Vector store is empty. Please add documents first.")
            
        docs = self.vector_store.similarity_search(query, k=k)
        
        # chonking
        docs.sort(key=lambda x: (
            x.metadata.get('source', ''),
            x.metadata.get('chunk', {}).get('index', 0)
        ))
        
        return docs

    def clear(self) -> None:
        """Clear the vector store"""
        self.vector_store = None