from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from typing import List, Dict, Any, Optional, Union
import copy
import pandas as pd
import numpy as np
import json

class VectorStore:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vector_store = None
        self.structured_store = None
        
        # Existing text splitters
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

    def add_structured_data(self, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Add structured data embeddings separately from text"""
        try:
            # Create descriptive texts for columns
            column_texts = []
            column_metadata = []
            
            for col in df.columns:
                # Basic column info
                col_type = str(df[col].dtype)
                unique_count = df[col].nunique()
                
                # Create descriptive text
                desc_text = f"Column {col} contains {col_type} data with {unique_count} unique values."
                
                # Additional statistical information for numeric columns
                if np.issubdtype(df[col].dtype, np.number):
                    desc_text += f" Range: {df[col].min()} to {df[col].max()}. "
                    desc_text += f"Mean: {df[col].mean():.2f}. "
                    desc_text += f"Median: {df[col].median()}"
                
                column_texts.append(desc_text)
                
                # Enhanced metadata
                col_metadata = {
                    **metadata,
                    'column_name': col,
                    'column_type': col_type,
                    'unique_values': unique_count,
                    'is_numeric': np.issubdtype(df[col].dtype, np.number),
                    'data_statistics': {
                        'min': float(df[col].min()) if np.issubdtype(df[col].dtype, np.number) else None,
                        'max': float(df[col].max()) if np.issubdtype(df[col].dtype, np.number) else None,
                        'mean': float(df[col].mean()) if np.issubdtype(df[col].dtype, np.number) else None,
                        'median': float(df[col].median()) if np.issubdtype(df[col].dtype, np.number) else None
                    }
                }
                column_metadata.append(col_metadata)

            # Store structured embeddings
            if self.structured_store is None:
                self.structured_store = FAISS.from_texts(
                    texts=column_texts,
                    embedding=self.embeddings,
                    metadatas=column_metadata
                )
            else:
                self.structured_store.add_texts(
                    texts=column_texts,
                    metadatas=column_metadata
                )

        except Exception as e:
            print(f"Error adding structured data: {str(e)}")
            raise

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the vector store with enhanced processing"""
        if not texts:
            return

        all_chunks = []
        all_metadatas = []

        for idx, text in enumerate(texts):
            # Get document type and appropriate splitter
            doc_type = 'default'
            if metadatas and idx < len(metadatas):
                doc_type = metadatas[idx].get('type', 'default')
            
            splitter = self._get_splitter(doc_type)
            chunks = splitter.split_text(text)

            # Handle metadata for chunks
            if metadatas and idx < len(metadatas):
                base_metadata = metadatas[idx]
                for chunk_idx, _ in enumerate(chunks):
                    chunk_metadata = self._enhance_metadata(
                        base_metadata,
                        chunk_idx,
                        len(chunks)
                    )
                    all_metadatas.append(chunk_metadata)
            else:
                for chunk_idx, _ in enumerate(chunks):
                    all_metadatas.append({
                        'chunk': {'index': chunk_idx, 'total': len(chunks)},
                        'type': 'default'
                    })

            all_chunks.extend(chunks)

        # Create or update vector store
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

    def similarity_search(self, query: str, k: int = 4, include_structured: bool = True) -> List[Dict[str, Any]]:
        """Enhanced search that includes structured data results"""
        results = []
        
        # Search regular document store
        if self.vector_store:
            results.extend(self.vector_store.similarity_search(query, k=k))
        
        # Search structured store if requested
        if include_structured and self.structured_store:
            structured_results = self.structured_store.similarity_search(query, k=k)
            results.extend(structured_results)
        
        # Sort results by relevance (if score is available)
        results.sort(
            key=lambda x: float(x.metadata.get('score', 0)) 
            if 'score' in x.metadata else 0,
            reverse=True
        )
        
        # Return top k results
        return results[:k]

    def clear(self) -> None:
        """Clear both regular and structured stores"""
        self.vector_store = None
        self.structured_store = None

    def get_structured_metadata(self) -> Dict[str, Any]:
        """Get metadata about stored structured data"""
        if not self.structured_store:
            return {}
            
        try:
            metadata = {}
            if hasattr(self.structured_store, 'docstore'):
                for doc_id in self.structured_store.docstore._dict:
                    doc = self.structured_store.docstore._dict[doc_id]
                    if hasattr(doc, 'metadata'):
                        source = doc.metadata.get('source', 'unknown')
                        if source not in metadata:
                            metadata[source] = {
                                'columns': [],
                                'numeric_columns': []
                            }
                        
                        col_name = doc.metadata.get('column_name')
                        if col_name:
                            metadata[source]['columns'].append(col_name)
                            if doc.metadata.get('is_numeric', False):
                                metadata[source]['numeric_columns'].append(col_name)
                                
            return metadata
        except Exception as e:
            print(f"Error getting structured metadata: {str(e)}")
            return {}