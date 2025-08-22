"""RAG indexing pipeline for codebases."""

import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .chunker import CodebaseChunker, CodeChunk
from .embeddings import CodeEmbedder
from .vector_store import ChromaVectorStore


@dataclass
class IndexingStats:
    total_files: int
    processed_files: int
    total_chunks: int
    skipped_files: int
    errors: int
    processing_time: float


class CodebaseIndexer:
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "codebase",
                 embedding_model: str = "",
                 ignore_patterns: Optional[List[str]] = None):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        print("Initializing embedder...")
        self.embedder = CodeEmbedder(model_name=embedding_model)
        
        print("Initializing vector store...")
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=self.embedder
        )
        
        self.chunker = CodebaseChunker(ignore_patterns=ignore_patterns)
        self.stats = None
    
    def index_codebase(self, codebase_path: str, clear_existing: bool = False, use_cache: bool = True) -> IndexingStats:
        start_time = time.time()
        
        if clear_existing:
            print("Clearing existing collection...")
            self.vector_store.clear_collection()
        
        print(f"Starting indexing of codebase: {codebase_path}")
        
        try:
            # Add progress callback for chunking
            print("🔍 Phase 1/3: Scanning and chunking files...")
            print()  # Add blank line for progress bar
            chunks = None
            try:
                chunks = self.chunker.chunk_codebase(codebase_path, progress_callback=self._chunking_progress, use_cache=use_cache)
            except KeyboardInterrupt:
                print("\n⚠️  Chunking interrupted by user!")
                print("💾 Partial progress has been saved.")
                raise
            
            if not chunks:
                print("No code chunks found to index.")
                return IndexingStats(0, 0, 0, 0, 0, time.time() - start_time)
            
            print(f"\n✅ Generated {len(chunks)} code chunks from {len(set(chunk.file_path for chunk in chunks))} files")
            
            # Display detailed chunk statistics
            # self._print_chunk_statistics(chunks)
            
            print("🧠 Phase 2/3: Generating embeddings and storing in vector database...")
            
            # Create a truly lazy filtering generator for size only
            max_chunk_size = 50000  # 5KB limit
            min_chunk_size = 25
            
            def filtered_chunks_generator():
                filtered = 0
                for chunk in chunks:
                    if min_chunk_size < len(chunk.content) <= max_chunk_size:
                        yield chunk
                    else:
                        filtered += 1
                
                if filtered > 0:
                    print(f"\n   📊 Filtered out {filtered:,} chunks larger than {max_chunk_size:,} characters")

            try:
                total_processed = self.vector_store.add_chunks_lazy(
                    filtered_chunks_generator(), 
                    progress_callback=self._embedding_progress
                )
            except KeyboardInterrupt:
                print("\n⚠️  Indexing interrupted by user!")
                print("💾 Partial progress has been saved.")
                raise
            
            print("📊 Phase 3/3: Finalizing...")
            
            processing_time = time.time() - start_time
            
            stats = IndexingStats(
                total_files=len(set(chunk.file_path for chunk in chunks)),
                processed_files=len(set(chunk.file_path for chunk in chunks)),
                total_chunks=total_processed,
                skipped_files=0,  # TODO: Track skipped files
                errors=0,  # TODO: Track errors
                processing_time=processing_time
            )
            
            self.stats = stats
            self._print_stats(stats)
            
            return stats
            
        except Exception as e:
            print(f"Error during indexing: {e}")
            raise
    
    def _chunking_progress(self, current: int, total: int, file_path: str = "", chunks_generated: int = 0):
        if total > 0:
            percent = (current / total) * 100
            bar_length = 30
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            remaining = total - current
            file_name = Path(file_path).name[:25] if file_path else ""
            print(f"\r   [{bar}] {percent:.1f}% | Files: {current}/{total} ({remaining} remaining) | Chunks: {chunks_generated} | {file_name}", end='', flush=True)
            
            # Force output frequently to ensure visibility on large codebases
            import sys
            if current == 0 or current == total or current % 3 == 0:
                sys.stdout.flush()  # Force flush the current line
    
    def _embedding_progress(self, current: int, total: int, phase: str = ""):
        if total is not None and total > 0:
            percent = (current / total) * 100
            bar_length = 30
            filled_length = int(bar_length * current // total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            print(f"\r   [{bar}] {percent:.1f}% ({current}/{total}) {phase}", end='', flush=True)
            if current == total:
                print()  # New line when complete
        else:
            # When total is None, just show current progress without percentage
            print(f"\r   Processing... {current} chunks {phase}", end='', flush=True)
    
    def _print_chunk_statistics(self, chunks):
        """Print detailed statistics about the chunks."""
        import numpy as np
        from collections import defaultdict
        from pathlib import Path
        
        if not chunks:
            print("   📊 No chunks to analyze")
            return
        
        print(f"\n📊 Chunk Statistics:")
        
        # Basic metrics
        chunk_lengths = [len(chunk.content) for chunk in chunks]
        chunk_lines = [chunk.end_line - chunk.start_line + 1 for chunk in chunks]
        
        # Overall statistics
        total_chars = sum(chunk_lengths)
        avg_length = np.mean(chunk_lengths)
        median_length = np.median(chunk_lengths)
        std_length = np.std(chunk_lengths)
        
        print(f"   Total chunks: {len(chunks):,}")
        print(f"   Total characters: {total_chars:,} ({total_chars/1024/1024:.1f} MB)")
        print(f"   Average length: {avg_length:.0f} characters")
        print(f"   Median length: {median_length:.0f} characters")
        print(f"   Std deviation: {std_length:.0f} characters")
        
        # Quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        print(f"\n   📈 Length Quantiles:")
        for q in quantiles:
            value = np.quantile(chunk_lengths, q)
            print(f"      {q*100:4.0f}%: {value:6.0f} chars")
        
        # Size distribution
        size_buckets = [
            (0, 100, "Tiny"),
            (100, 500, "Small"), 
            (500, 1000, "Medium"),
            (1000, 5000, "Large"),
            (5000, 10000, "Very Large"),
            (10000, float('inf'), "Huge")
        ]
        
        print(f"\n   📊 Size Distribution:")
        for min_size, max_size, label in size_buckets:
            count = len([c for c in chunk_lengths if min_size <= c < max_size])
            pct = (count / len(chunks)) * 100
            if count > 0:
                print(f"      {label:10s}: {count:5,} chunks ({pct:4.1f}%)")
        
        # By chunk type
        type_stats = defaultdict(list)
        for chunk in chunks:
            type_stats[chunk.chunk_type].append(len(chunk.content))
        
        print(f"\n   🏷️  By Chunk Type:")
        for chunk_type, lengths in sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(lengths)
            avg_len = np.mean(lengths)
            median_len = np.median(lengths)
            pct = (count / len(chunks)) * 100
            print(f"      {chunk_type:12s}: {count:5,} chunks ({pct:4.1f}%) | avg: {avg_len:5.0f} | median: {median_len:5.0f}")
        
        # By file extension
        ext_stats = defaultdict(list)
        for chunk in chunks:
            ext = Path(chunk.file_path).suffix.lower() or 'no_ext'
            ext_stats[ext].append(len(chunk.content))
        
        print(f"\n   📁 By File Extension (top 10):")
        sorted_exts = sorted(ext_stats.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for ext, lengths in sorted_exts:
            count = len(lengths)
            avg_len = np.mean(lengths)
            pct = (count / len(chunks)) * 100
            print(f"      {ext:12s}: {count:5,} chunks ({pct:4.1f}%) | avg: {avg_len:5.0f}")
        
        # Line count statistics
        avg_lines = np.mean(chunk_lines)
        median_lines = np.median(chunk_lines)
        
        print(f"\n   📏 Line Statistics:")
        print(f"      Average lines per chunk: {avg_lines:.1f}")
        print(f"      Median lines per chunk: {median_lines:.1f}")
        
        # Identify potential issues
        print(f"\n   ⚠️  Potential Issues:")
        empty_chunks = len([c for c in chunk_lengths if c == 0])
        tiny_chunks = len([c for c in chunk_lengths if 0 < c < 50])
        huge_chunks = len([c for c in chunk_lengths if c > 10000])
        
        if empty_chunks > 0:
            print(f"      Empty chunks: {empty_chunks}")
        if tiny_chunks > 0:
            print(f"      Very tiny chunks (<50 chars): {tiny_chunks}")
        if huge_chunks > 0:
            print(f"      Very large chunks (>10K chars): {huge_chunks}")
        
        if empty_chunks == 0 and tiny_chunks == 0 and huge_chunks == 0:
            print(f"      ✅ No obvious issues detected")

    def update_file(self, file_path: str, codebase_root: str):
        try:
            file_chunks = []
            
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                chunks = self.chunker.tree_sitter_chunker.chunk_code(content, file_path)
                file_chunks.extend(chunks)
            
            self.vector_store.update_chunks(file_chunks)
            print(f"Updated indexing for: {file_path}")
            
        except Exception as e:
            print(f"Error updating file {file_path}: {e}")
    
    def search_code(self, query: str, n_results: int = 10, diversity: float = 0.0, **kwargs) -> List[Dict[str, Any]]:
        if diversity > 0:
            return self.vector_store.search_with_diversity(query, n_results, diversity, **kwargs)
        else:
            return self.vector_store.search(query, n_results, **kwargs)
    
    def search_by_language(self, query: str, languages: List[str], n_results: int = 10) -> List[Dict[str, Any]]:
        extensions = [f".{lang}" if not lang.startswith('.') else lang for lang in languages]
        return self.vector_store.search_by_file_type(query, extensions, n_results)
    
    def search_functions(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        return self.vector_store.search_by_chunk_type(query, ['function', 'method'], n_results)
    
    def search_classes(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        return self.vector_store.search_by_chunk_type(query, ['class'], n_results)
    
    def get_collection_info(self) -> Dict[str, Any]:
        info = self.vector_store.get_collection_info()
        if self.stats:
            info.update({
                'last_indexing_stats': {
                    'total_files': self.stats.total_files,
                    'total_chunks': self.stats.total_chunks,
                    'processing_time': f"{self.stats.processing_time:.2f}s"
                }
            })
        return info
    
    def export_index(self, output_path: str):
        self.vector_store.export_data(output_path)
    
    def get_file_coverage(self) -> Dict[str, int]:
        return self.vector_store.get_file_coverage()
    
    def clear_chunk_cache(self, codebase_path: str) -> bool:
        """Clear the chunk cache for a specific codebase."""
        return self.chunker.clear_chunks_cache(codebase_path)
    
    def _print_stats(self, stats: IndexingStats):
        print("\n" + "="*50)
        print("INDEXING COMPLETE")
        print("="*50)
        print(f"Total files processed: {stats.processed_files}")
        print(f"Total code chunks: {stats.total_chunks}")
        print(f"Processing time: {stats.processing_time:.2f} seconds")
        print(f"Average chunks per file: {stats.total_chunks / max(stats.processed_files, 1):.1f}")
        print(f"Database location: {self.persist_directory}")
        print("="*50)


class IncrementalIndexer:
    def __init__(self, indexer: CodebaseIndexer):
        self.indexer = indexer
        self.file_timestamps = {}
        self._load_timestamps()
    
    def _load_timestamps(self):
        timestamp_file = Path(self.indexer.persist_directory) / "file_timestamps.json"
        if timestamp_file.exists():
            import json
            with open(timestamp_file, 'r') as f:
                self.file_timestamps = json.load(f)
    
    def _save_timestamps(self):
        timestamp_file = Path(self.indexer.persist_directory) / "file_timestamps.json"
        import json
        with open(timestamp_file, 'w') as f:
            json.dump(self.file_timestamps, f)
    
    def update_if_changed(self, file_path: str, codebase_root: str):
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            if file_path in self.file_timestamps:
                self.indexer.vector_store.delete_by_file_path(file_path)
                del self.file_timestamps[file_path]
                self._save_timestamps()
            return
        
        current_mtime = file_path_obj.stat().st_mtime
        last_mtime = self.file_timestamps.get(file_path, 0)
        
        if current_mtime > last_mtime:
            self.indexer.update_file(file_path, codebase_root)
            self.file_timestamps[file_path] = current_mtime
            self._save_timestamps()
    
    def incremental_update(self, codebase_path: str):
        print(f"Performing incremental update for: {codebase_path}")
        
        codebase_root = Path(codebase_path)
        updated_files = 0
        
        # Set up gitignore matcher
        gitignore_matcher = None
        gitignore_path = codebase_root / '.gitignore'
        if gitignore_path.exists():
            import gitignore_parser
            gitignore_matcher = gitignore_parser.parse_gitignore(str(gitignore_path))
        
        for file_path in codebase_root.rglob('*'):
            if file_path.is_file() and self.indexer.chunker.should_process_file(file_path, gitignore_matcher, codebase_root):
                self.update_if_changed(str(file_path), codebase_path)
                updated_files += 1
        
        print(f"Incremental update complete. Checked {updated_files} files.")
