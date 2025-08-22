"""Local embedding model integration for code chunks."""

import os
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path


class LocalEmbedder:
    def __init__(self, model_name: str = "", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "codebase_rag")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading embedding model: {self.model_name}")
        
        self.model = SentenceTransformer(
            self.model_name,
            # model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "trust_remote_code": True},
            # model_kwargs={"trust_remote_code": True},
            # tokenizer_kwargs={"padding_side": "left"},
            cache_folder=self.cache_dir,
            trust_remote_code=True,
            device='mps',
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query with 'query:' prefix for E5 models"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Add query prefix for E5 models
        prefixed_query = f"query: {query}"
        embeddings = self.model.encode([prefixed_query], convert_to_numpy=True, show_progress_bar=False)
        return embeddings[0]
    
    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:
        """Embed documents with 'passage:' prefix for E5 models"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        # Add passage prefix for E5 models
        prefixed_documents = [f"passage: {doc}" for doc in documents]
        
        # Use batching and disable progress bar for speed
        embeddings = self.model.encode(
            prefixed_documents, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=16,  # Optimize batch size for MPS
            normalize_embeddings=True  # Normalize for better performance
        )
        return embeddings
    
    def embed_text(self, text: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generic text embedding - uses passage prefix by default"""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        if isinstance(text, str):
            prefixed_text = f"passage: {text}"
            embeddings = self.model.encode([prefixed_text], convert_to_numpy=True, show_progress_bar=True)
            return embeddings[0]
        else:
            prefixed_texts = [f"passage: {t}" for t in text]
            embeddings = self.model.encode(prefixed_texts, convert_to_numpy=True, show_progress_bar=True)
            return embeddings
    
    def embed_code_chunks(self, chunks: List) -> List[np.ndarray]:
        texts = []
        for chunk in chunks:
            text = f"{chunk.metadata.get('name', '')}\n{chunk.content}"
            if chunk.chunk_type != 'text_block':
                text = f"[{chunk.chunk_type}] {text}"
            texts.append(text)
        
        return self.embed_text(texts)
    
    def get_embedding_dimension(self) -> int:
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()


class CodeEmbedder(LocalEmbedder):
    def __init__(self, model_name: str = "", cache_dir: Optional[str] = None):
        super().__init__(model_name, cache_dir)
    
    def prepare_code_text(self, chunk) -> str:
        language = chunk.metadata.get('language', '').lstrip('.')
        name = chunk.metadata.get('name', 'anonymous')
        chunk_type = chunk.chunk_type
        
        prefix = f"[{language}] [{chunk_type}] {name}:"
        
        content = chunk.content.strip()
        if len(content) > 8000:  # Truncate very long chunks
            content = content[:8000] + "..."
        
        # Add passage prefix for E5 models
        return f"passage: {prefix}\n{content}"
    
    def embed_code_chunks(self, chunks: List) -> List[np.ndarray]:
        """Embed code chunks - already prefixed with 'passage:' in prepare_code_text"""
        texts = [self.prepare_code_text(chunk) for chunk in chunks]
        # Don't use embed_documents since we already added the prefix
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            batch_size=16,
            normalize_embeddings=True
        )
        return embeddings
    
    def embed_code_query(self, query: str) -> np.ndarray:
        """Embed a code search query with 'query:' prefix"""
        return self.embed_query(query)
