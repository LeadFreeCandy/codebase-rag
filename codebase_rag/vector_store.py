"""Chroma database integration for vector storage."""

import os
import json
import uuid
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

from .chunker import CodeChunk
from .embeddings import CodeEmbedder


class ChromaVectorStore:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "codebase",
                 embedder: Optional[CodeEmbedder] = None):
        
        self.persist_directory = Path(persist_directory).absolute()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self.embedder = embedder or CodeEmbedder()
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = None
        self._setup_collection()
    
    def _setup_collection(self):
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Set environment variables for trust_remote_code
            import os
            os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"
            os.environ["TRUST_REMOTE_CODE"] = "1"
            
            # Create custom embedding function that supports trust_remote_code
            class TrustedEmbeddingFunction(embedding_functions.SentenceTransformerEmbeddingFunction):
                def __init__(self, model_name, embedder=None, **kwargs):
                    self.embedder = embedder
                    
                    # Pre-load model components with trust_remote_code
                    from transformers import AutoConfig, AutoTokenizer, AutoModel
                    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True) 
                    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                    
                    print("loading model here!!!")
                    # Initialize with trust_remote_code
                    super().__init__(device='mps', model_name=model_name, trust_remote_code=True, **kwargs)
                
                def __call__(self, input):
                    """Use batched embedding for much better performance"""
                    if self.embedder and hasattr(self.embedder, 'embed_documents'):
                        if isinstance(input, list):
                            # Check if this is a query (single item list from search) or document batch
                            if len(input) == 1 and not input[0].startswith("passage:"):
                                # This is a search query - use query prefix
                                embedding = self.embedder.embed_code_query(input[0])
                                return [embedding.tolist()]
                            else:
                                # This is document indexing - use document embedding (already has passage: prefix)
                                embeddings = self.embedder.embed_documents(input)
                                return [emb.tolist() for emb in embeddings]
                        else:
                            # Single non-list input - use query method
                            embedding = self.embedder.embed_code_query(input)
                            return embedding.tolist()
                    else:
                        return super().__call__(input)
            
            embedding_function = TrustedEmbeddingFunction(
                model_name=self.embedder.model_name,
                embedder=self.embedder
            )
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {self.collection_name}")
    
    def add_chunks_lazy(self, chunks_generator, batch_size: int = 16, progress_callback=None):
        """
        Add chunks from a generator to avoid loading all chunks into memory at once.
        """
        batch = []
        total_processed = 0
        batch_num = 1
        
        try:
            for chunk in chunks_generator:
                batch.append(chunk)
                
                # Process batch when it's full
                if len(batch) >= batch_size:
                    self._add_batch(batch)
                    total_processed += len(batch)
                    
                    if progress_callback:
                        progress_callback(total_processed, None, f"Embedding batch {batch_num}")
                    else:
                        print(f"Processed batch {batch_num} ({total_processed} total chunks)")
                    
                    batch = []
                    batch_num += 1
                    
                    # Force garbage collection after every few batches to free memory
                    if batch_num % 10 == 0:  # More frequent GC
                        import gc
                        gc.collect()
            
            # Process remaining chunks in the last batch
            if batch:
                self._add_batch(batch)
                total_processed += len(batch)
                
                if progress_callback:
                    progress_callback(total_processed, None, f"Final batch {batch_num}")
                else:
                    print(f"Processed final batch {batch_num} ({total_processed} total chunks)")
                    
        except KeyboardInterrupt:
            print(f"\n⚠️  Embedding interrupted! Processed {total_processed} chunks.")
            print("💾 Partial embeddings have been saved to the database.")
            raise
        
        print(f"✅ Successfully processed {total_processed} chunks in {batch_num} batches")
        return total_processed

    def add_chunks(self, chunks: List[CodeChunk], batch_size: int = 15, progress_callback=None):
        if not chunks:
            return
        
        total_chunks = len(chunks)
        
        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                self._add_batch(batch)
                
                processed = min(i + batch_size, len(chunks))
                if progress_callback:
                    progress_callback(processed, total_chunks, f"Embedding batch {i//batch_size + 1}")
                else:
                    print(f"Processed {processed}/{total_chunks} chunks")
                
                # Force garbage collection after every few batches to free memory
                if (i // batch_size + 1) % 20 == 0:
                    import gc
                    gc.collect()
                    
        except KeyboardInterrupt:
            processed = min(i, len(chunks))
            print(f"\n⚠️  Embedding interrupted! Processed {processed}/{total_chunks} chunks.")
            print("💾 Partial embeddings have been saved to the database.")
            raise
    
    def _add_batch(self, chunks: List[CodeChunk]):
        documents = []
        metadatas = []
        ids = []
        used_ids = set()
        
        try:
            for i, chunk in enumerate(chunks):
                base_id = self._generate_chunk_id(chunk)
                chunk_id = base_id
                
                # Ensure uniqueness by adding counter if needed
                counter = 0
                while chunk_id in used_ids:
                    counter += 1
                    chunk_id = f"{base_id}#{counter}"
                used_ids.add(chunk_id)
                
                documents.append(chunk.content)
                
                metadata = {
                    'file_path': chunk.file_path,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'chunk_type': chunk.chunk_type,
                    **chunk.metadata
                }
                metadatas.append(metadata)
                ids.append(chunk_id)
            
            # Pre-compute embeddings in batch for massive speedup
            embeddings = self.embedder.embed_code_chunks(chunks)
            
            # Convert to list format for ChromaDB
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            # Add to collection with pre-computed embeddings
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings_list
            )
        except KeyboardInterrupt:
            # Re-raise to be handled at the higher level
            raise
    
    def _generate_chunk_id(self, chunk: CodeChunk) -> str:
        import hashlib
        # Use full content hash for uniqueness and add chunk type
        content_hash = hashlib.md5(chunk.content.encode('utf-8')).hexdigest()[:12]
        return f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.chunk_type}:{content_hash}"
    
    def search(self, query: str, n_results: int = 10, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        search_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            }
            search_results.append(result)
        
        return search_results
    
    def search_by_file_type(self, query: str, file_extensions: List[str], n_results: int = 10) -> List[Dict[str, Any]]:
        where_clause = {"language": {"$in": file_extensions}}
        return self.search(query, n_results, where_clause)
    
    def search_by_chunk_type(self, query: str, chunk_types: List[str], n_results: int = 10) -> List[Dict[str, Any]]:
        where_clause = {"chunk_type": {"$in": chunk_types}}
        return self.search(query, n_results, where_clause)
    
    def search_with_diversity(self, query: str, n_results: int = 10, diversity: float = 0.5, 
                            where: Optional[Dict] = None, candidate_factor: int = 3) -> List[Dict[str, Any]]:
        """
        Search with Maximal Marginal Relevance (MMR) for diversity.
        
        Args:
            query: Search query
            n_results: Number of final results to return
            diversity: Trade-off between relevance and diversity (0.0 = pure relevance, 1.0 = pure diversity)
            where: Optional where clause for filtering
            candidate_factor: Factor to multiply n_results for initial candidate retrieval
        
        Returns:
            List of diverse search results
        """
        if diversity <= 0:
            # If no diversity requested, use regular search
            return self.search(query, n_results, where)
        
        # Get more candidates than needed for MMR selection
        n_candidates = min(n_results * candidate_factor, 100)  # Cap at 100 to avoid excessive computation
        
        # Get initial candidates
        candidates = self.collection.query(
            query_texts=[query],
            n_results=n_candidates,
            where=where,
            include=['embeddings', 'documents', 'metadatas', 'distances']
        )
        
        if not candidates['ids'][0]:
            return []
        
        # Get query embedding for MMR calculation
        query_embedding = self.embedder.embed_code_query(query)
        
        # Convert to numpy arrays for easier computation
        candidate_embeddings = np.array(candidates['embeddings'][0])
        candidate_distances = np.array(candidates['distances'][0])
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(candidates['ids'][0])))
        
        # Select first document (highest relevance)
        best_idx = np.argmin(candidate_distances)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Select remaining documents using MMR
        while len(selected_indices) < n_results and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score (similarity to query, converted from distance)
                relevance = 1 - candidate_distances[idx]
                
                # Diversity score (minimum similarity to already selected documents)
                if selected_indices:
                    selected_embeddings = candidate_embeddings[selected_indices]
                    similarities_to_selected = np.dot(candidate_embeddings[idx], selected_embeddings.T)
                    max_similarity_to_selected = np.max(similarities_to_selected)
                    diversity_score = 1 - max_similarity_to_selected
                else:
                    diversity_score = 1.0
                
                # MMR score combines relevance and diversity
                mmr_score = (1 - diversity) * relevance + diversity * diversity_score
                mmr_scores.append((mmr_score, idx))
            
            # Select document with highest MMR score
            best_mmr_score, best_idx = max(mmr_scores)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Build result list
        search_results = []
        for idx in selected_indices:
            result = {
                'id': candidates['ids'][0][idx],
                'document': candidates['documents'][0][idx],
                'metadata': candidates['metadatas'][0][idx],
                'distance': candidates['distances'][0][idx]
            }
            search_results.append(result)
        
        return search_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': str(self.persist_directory)
        }
    
    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self._setup_collection()
        print(f"Cleared collection: {self.collection_name}")
    
    def delete_by_file_path(self, file_path: str):
        try:
            self.collection.delete(
                where={"file_path": file_path}
            )
            print(f"Deleted chunks for file: {file_path}")
        except Exception as e:
            print(f"Error deleting chunks for {file_path}: {e}")
    
    def update_chunks(self, chunks: List[CodeChunk]):
        if not chunks:
            return
        
        file_paths = list(set(chunk.file_path for chunk in chunks))
        
        for file_path in file_paths:
            self.delete_by_file_path(file_path)
        
        file_chunks = [chunk for chunk in chunks if chunk.file_path in file_paths]
        self.add_chunks(file_chunks)
        
        print(f"Updated {len(file_chunks)} chunks for {len(file_paths)} files")
    
    def export_data(self, output_path: str):
        results = self.collection.get()
        
        export_data = {
            'collection_name': self.collection_name,
            'count': len(results['ids']),
            'chunks': []
        }
        
        for i in range(len(results['ids'])):
            chunk_data = {
                'id': results['ids'][i],
                'document': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            export_data['chunks'].append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"Exported {export_data['count']} chunks to {output_path}")
    
    def get_file_coverage(self) -> Dict[str, int]:
        results = self.collection.get()
        file_coverage = {}
        
        for metadata in results['metadatas']:
            file_path = metadata['file_path']
            file_coverage[file_path] = file_coverage.get(file_path, 0) + 1
        
        return file_coverage
