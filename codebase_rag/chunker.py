"""Code chunking functionality for different file types."""

import os
import re
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import tree_sitter
from tree_sitter import Language, Parser
import gitignore_parser


@dataclass
class CodeChunk:
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, method, module, etc.
    metadata: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "chunk_type": self.chunk_type,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeChunk':
        """Create from dictionary (JSON deserialization)."""
        return cls(
            content=data["content"],
            file_path=data["file_path"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            chunk_type=data["chunk_type"],
            metadata=data["metadata"]
        )


class TreeSitterChunker:
    def __init__(self):
        self.parsers = {}
        self.languages = {}
        self._setup_languages()
    
    def _setup_languages(self):
        language_configs = {
            'python': ('tree_sitter_python', 'language', ['.py']),
            'javascript': ('tree_sitter_javascript', 'language', ['.js', '.jsx']),
            'typescript': ('tree_sitter_typescript', 'language_typescript', ['.ts']),
            'tsx': ('tree_sitter_typescript', 'language_tsx', ['.tsx']),
            'go': ('tree_sitter_go', 'language', ['.go']),
            'rust': ('tree_sitter_rust', 'language', ['.rs']),
            'java': ('tree_sitter_java', 'language', ['.java']),
            'cpp': ('tree_sitter_cpp', 'language', ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp']),
            'ruby': ('tree_sitter_ruby', 'language', ['.rb'])
        }
        
        for lang_name, (module_name, lang_func, extensions) in language_configs.items():
            try:
                module = __import__(module_name)
                lang_function = getattr(module, lang_func)
                language = Language(lang_function())
                parser = Parser(language)
                
                self.languages[lang_name] = language
                self.parsers[lang_name] = parser
                
                for ext in extensions:
                    self.parsers[ext] = parser
                    self.languages[ext] = language
            except (ImportError, AttributeError) as e:
                print(f"Warning: {module_name}.{lang_func} not available, skipping {lang_name} support: {e}")
    
    def get_language_for_file(self, file_path: str) -> Optional[str]:
        ext = Path(file_path).suffix.lower()
        return ext if ext in self.parsers else None
    
    def chunk_code(self, content: str, file_path: str) -> List[CodeChunk]:
        lang = self.get_language_for_file(file_path)
        if not lang or lang not in self.parsers:
            return self._fallback_chunk(content, file_path)
        
        parser = self.parsers[lang]
        tree = parser.parse(bytes(content, 'utf8'))
        
        chunks = []
        self._extract_chunks(tree.root_node, content, file_path, chunks)
        
        # Add a module-level chunk representing the entire file
        lines = content.split('\n')
        module_chunk = CodeChunk(
            content=content,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            chunk_type='module',
            metadata={
                'name': Path(file_path).stem,
                'language': lang or 'unknown',
                'node_type': 'module'
            }
        )
        chunks.append(module_chunk)
        
        if len(chunks) == 1:  # Only the module chunk was added (no individual constructs found)
            return self._fallback_chunk(content, file_path)
        
        return chunks
    
    def _extract_chunks(self, node, content: str, file_path: str, chunks: List[CodeChunk]):
        lines = content.split('\n')
        
        chunk_types = {
            'function_definition': 'function',
            'method_definition': 'method',
            'class_definition': 'class',
            'function_declaration': 'function',
            'method_declaration': 'method',
            'class_declaration': 'class',
            'interface_declaration': 'interface',
            'type_declaration': 'type',
            'struct_item': 'struct',
            'impl_item': 'impl',
            'enum_item': 'enum',
            'trait_item': 'trait',
            'method': 'method',
            'class': 'class',
            'module': 'module'
        }
        
        if node.type in chunk_types:
            start_line = node.start_point[0]
            end_line = node.end_point[0]
            
            chunk_content = '\n'.join(lines[start_line:end_line + 1])
            
            # Skip nodes that are just keywords or single lines without meaningful content
            # These are typically child nodes that represent just the declaration
            if self._is_meaningful_chunk(node, chunk_content):
                name = self._extract_name(node, content)
                metadata = {
                    'name': name,
                    'language': self.get_language_for_file(file_path) or 'unknown',
                    'node_type': node.type
                }
                
                chunk = CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line + 1,
                    end_line=end_line + 1,
                    chunk_type=chunk_types[node.type],
                    metadata=metadata
                )
                chunks.append(chunk)
        
        for child in node.children:
            self._extract_chunks(child, content, file_path, chunks)
    
    def _is_meaningful_chunk(self, node, chunk_content: str) -> bool:
        """
        Determine if a chunk is meaningful enough to include.
        Filters out nodes that are just keywords or single-line declarations.
        """
        # Skip very short content (likely just keywords)
        stripped_content = chunk_content.strip()
        if len(stripped_content) < 10:
            return False
        
        # Count non-empty lines
        lines = [line.strip() for line in stripped_content.split('\n') if line.strip()]
        
        # For modules and classes, we want chunks with actual content, not just declarations
        if node.type in ['module', 'class']:
            # Must have more than just the declaration line
            if len(lines) <= 1:
                return False
            
            # Should have some body content (methods, variables, etc.)
            # Look for common patterns that indicate real content
            content_lower = stripped_content.lower()
            has_content_indicators = any(indicator in content_lower for indicator in [
                'def ', 'class ', 'module ', '@', 'attr_', 'include ', 'extend ',
                'private', 'protected', 'public', '=', 'if ', 'when ', 'case ',
                'begin', 'rescue', 'ensure', 'end'
            ])
            
            # If it's a module/class but has no content indicators and is very short, skip it
            if not has_content_indicators and len(stripped_content) < 50:
                return False
        
        # For methods, we're more lenient as even short methods can be meaningful
        if node.type in ['method', 'function', 'method_definition', 'function_definition']:
            # Methods should have at least 2 lines (def + end or similar)
            return len(lines) >= 2
        
        return True
    
    def _extract_name(self, node, content: str) -> str:
        for child in node.children:
            if child.type in ['identifier', 'type_identifier', 'field_identifier']:
                return content[child.start_byte:child.end_byte]
        return f"anonymous_{node.type}"
    
    def _fallback_chunk(self, content: str, file_path: str) -> List[CodeChunk]:
        lines = content.split('\n')
        max_lines = 100
        chunks = []
        
        for i in range(0, len(lines), max_lines):
            end_idx = min(i + max_lines, len(lines))
            chunk_content = '\n'.join(lines[i:end_idx])
            
            chunk = CodeChunk(
                content=chunk_content,
                file_path=file_path,
                start_line=i + 1,
                end_line=end_idx,
                chunk_type='text_block',
                metadata={
                    'language': Path(file_path).suffix.lower() or 'unknown',
                    'fallback': 'true'
                }
            )
            chunks.append(chunk)
        
        return chunks


class CodebaseChunker:
    def __init__(self, ignore_patterns: Optional[List[str]] = None):
        self.tree_sitter_chunker = TreeSitterChunker()
        self.ignore_patterns = ignore_patterns or [
            '.git/', '__pycache__/', 'node_modules/', '.venv/',
            '*.pyc', '*.pyo', '*.so', '*.dll', '*.exe',
            '*.jpg', '*.png', '*.gif', '*.pdf', '*.zip'
        ]
        self.interrupted = False
    
    
    def _get_git_files(self, root: Path) -> List[Path]:
        """Get list of files from git that should be processed."""
        import subprocess
        
        try:
            # Change to the repository directory
            original_cwd = Path.cwd()
            os.chdir(root)
            
            # Get untracked and tracked files from git
            cmd = "(git status --short | grep '^?' | cut -d\\  -f2- && git ls-files) | sort -u"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("   ⚠️  Not a git repository, falling back to basic file discovery")
                # Fallback to simple file discovery
                files = []
                for f in root.rglob('*'):
                    if f.is_file() and not self._should_skip_basic(f):
                        files.append(f)
                return files
            
            files = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    file_path = root / line.strip()
                    if file_path.exists() and file_path.is_file():
                        # Check size limit (2MB)
                        if file_path.stat().st_size <= 1 * 100 * 1024:
                            files.append(file_path)
                        else:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"      📏 {file_path.relative_to(root)} (too large: {size_mb:.1f}MB)")
            
            return files
            
        except Exception as e:
            print(f"   ⚠️  Git command failed: {e}, using basic file discovery")
            files = []
            for f in root.rglob('*'):
                if f.is_file() and not self._should_skip_basic(f):
                    files.append(f)
            return files
        finally:
            os.chdir(original_cwd)
    
    def _should_skip_basic(self, file_path: Path) -> bool:
        """Basic file filtering for non-git repositories."""
        # Skip very basic patterns
        basic_ignore = ['.git/', '__pycache__/', 'node_modules/', '.DS_Store']
        
        for pattern in basic_ignore:
            if pattern in str(file_path):
                return True
        
        # Skip files larger than 2MB
        if file_path.stat().st_size > 2 * 1024 * 1024:
            return True
            
        return False
    
    
    def _get_cache_path(self, root_path: str) -> Path:
        """Generate cache file path based on codebase path."""
        # Create a hash of the root path to ensure unique cache files
        path_hash = hashlib.md5(str(Path(root_path).resolve()).encode()).hexdigest()[:8]
        cache_dir = Path.home() / ".cache" / "codebase_rag" / "chunks"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        codebase_name = Path(root_path).name
        return cache_dir / f"{codebase_name}_{path_hash}_chunks.json"
    
    def save_chunks_cache(self, chunks: List[CodeChunk], root_path: str) -> None:
        """Save chunks to cache file in human-readable JSON format."""
        cache_path = self._get_cache_path(root_path)
        
        try:
            cache_data = {
                "metadata": {
                    "root_path": str(Path(root_path).resolve()),
                    "total_chunks": len(chunks),
                    "cache_version": "1.0",
                    "created_at": str(Path().cwd()),  # Current working directory as timestamp alternative
                },
                "chunks": [chunk.to_dict() for chunk in chunks]
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"   💾 Saved {len(chunks)} chunks to cache: {cache_path}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save chunk cache: {e}")
    
    def load_chunks_cache(self, root_path: str) -> Optional[List[CodeChunk]]:
        """Load chunks from cache file if it exists."""
        cache_path = self._get_cache_path(root_path)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Validate cache format
            if "chunks" not in cache_data or "metadata" not in cache_data:
                print(f"   ⚠️  Invalid cache format in {cache_path}")
                return None
            
            chunks = [CodeChunk.from_dict(chunk_data) for chunk_data in cache_data["chunks"]]
            
            metadata = cache_data["metadata"]
            print(f"   📂 Loaded {len(chunks)} chunks from cache ({metadata.get('total_chunks', 'unknown')} total)")
            print(f"      Cache: {cache_path}")
            
            return chunks
            
        except Exception as e:
            print(f"   ⚠️  Failed to load chunk cache: {e}")
            return None
    
    def clear_chunks_cache(self, root_path: str) -> bool:
        """Clear the chunk cache for a specific codebase."""
        cache_path = self._get_cache_path(root_path)
        
        try:
            if cache_path.exists():
                cache_path.unlink()
                print(f"   🗑️  Cleared chunk cache: {cache_path}")
                return True
            else:
                print(f"   📂 No cache file found to clear")
                return False
        except Exception as e:
            print(f"   ⚠️  Failed to clear cache: {e}")
            return False
    
    def chunk_codebase_lazy(self, root_path: str, progress_callback=None, use_cache: bool = True, max_chunk_size: int = 5000):
        """
        Generator version that yields chunks one at a time to avoid loading all chunks into memory.
        Filters out chunks larger than max_chunk_size automatically.
        """
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        
        # Try to load from cache first and yield from cache if available
        if use_cache:
            cached_chunks = self.load_chunks_cache(root_path)
            if cached_chunks is not None:
                print(f"   🔀 Using cached chunks with lazy loading and filtering...")
                filtered_count = 0
                for chunk in cached_chunks:
                    if len(chunk.content) <= max_chunk_size:
                        yield chunk
                    else:
                        filtered_count += 1
                if filtered_count > 0:
                    print(f"   ⚠️  Filtered out {filtered_count} cached chunks larger than {max_chunk_size:,} characters")
                return
        
        # No cache available, process files on-the-fly
        print("   📁 Getting files from git...")
        files_to_process = self._get_git_files(root)
        
        total_files = len(files_to_process)
        print(f"   📊 Found {total_files} files from git")
        
        if total_files == 0:
            print("   ⚠️  No files found to process")
            return
        
        processed_chunks = 0
        filtered_count = 0
        
        # Show initial progress
        if progress_callback:
            progress_callback(0, total_files, "Starting...", 0)
        
        for i, file_path in enumerate(files_to_process):
            try:
                relative_path = file_path.relative_to(root)
                
                # Show progress at start of file
                if progress_callback:
                    progress_callback(i, total_files, str(relative_path), processed_chunks)
                
                # Read and process file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Chunk the file
                file_chunks = self.tree_sitter_chunker.chunk_code(content, str(file_path))
                
                # Yield chunks that meet size requirements
                for chunk in file_chunks:
                    if len(chunk.content) <= max_chunk_size:
                        yield chunk
                        processed_chunks += 1
                    else:
                        filtered_count += 1
                        
            except Exception as e:
                print(f"   ⚠️  Error processing {file_path}: {e}")
                continue
        
        if filtered_count > 0:
            print(f"   ⚠️  Filtered out {filtered_count} chunks larger than {max_chunk_size:,} characters during processing")
        print(f"   ✅ Processed {processed_chunks} chunks from {total_files} files")

    def chunk_codebase(self, root_path: str, progress_callback=None, use_cache: bool = True) -> List[CodeChunk]:
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        
        # Try to load from cache first
        if use_cache:
            cached_chunks = self.load_chunks_cache(root_path)
            if cached_chunks is not None:
                return cached_chunks
        
        print("   📁 Getting files from git...")
        files_to_process = self._get_git_files(root)
        
        all_chunks = []
        total_files = len(files_to_process)
        skipped_size = 0
        
        print(f"   📊 Found {total_files} files from git")
        
        if total_files == 0:
            print("   ⚠️  No files found to process")
            return all_chunks
        
        # Show initial progress
        if progress_callback:
            progress_callback(0, total_files, "Starting...", 0)
        
        for i, file_path in enumerate(files_to_process):
            
            try:
                relative_path = file_path.relative_to(root)
                
                # Show progress at start of file
                if progress_callback:
                    progress_callback(i, total_files, str(relative_path), len(all_chunks))
                
                # Read and process file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                # Update progress before chunking
                if progress_callback:
                    file_size_kb = len(content) // 1024
                    progress_callback(i, total_files, f"{relative_path} ({file_size_kb}KB) - chunking...", len(all_chunks))
                
                # Chunk the code
                try:
                    chunks = self.tree_sitter_chunker.chunk_code(content, str(file_path))
                    all_chunks.extend(chunks)
                        
                except Exception as chunk_error:
                    print(f"\\nWarning: Failed to chunk {relative_path}: {chunk_error}")
                    # Create a fallback chunk for the entire file
                    fallback_chunks = self.tree_sitter_chunker._fallback_chunk(content, str(file_path))
                    all_chunks.extend(fallback_chunks)
                    chunks = fallback_chunks
                
                # Update progress after processing
                if progress_callback:
                    progress_callback(i + 1, total_files, f"{relative_path} - done (+{len(chunks)} chunks)", len(all_chunks))
                
            except Exception as e:
                print(f"\\nWarning: Failed to process {relative_path}: {e}")
                continue
        
        # Final progress update
        if progress_callback:
            progress_callback(total_files, total_files, "", len(all_chunks))
            print()  # New line after progress bar
        
        # Shuffle chunks for better speed estimates during processing
        if all_chunks:
            random.shuffle(all_chunks)
            print(f"   🔀 Shuffled {len(all_chunks)} chunks for better processing speed estimates")
        
        # Save to cache if we have chunks and caching is enabled
        if use_cache and all_chunks:
            self.save_chunks_cache(all_chunks, root_path)
        
        return all_chunks
