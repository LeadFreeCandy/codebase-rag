#!/usr/bin/env python3
"""Fixed MCP server for codebase RAG using FastMCP."""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from mcp.server.fastmcp import FastMCP

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from codebase_rag.indexer import CodebaseIndexer

# Create the MCP server
mcp = FastMCP("codebase-rag")

# Global indexer instance (lazy loaded)
_indexer: Optional[CodebaseIndexer] = None

def get_indexer() -> CodebaseIndexer:
    """Get or create the indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = CodebaseIndexer(
            persist_directory="./apm_bundle_full_db",
            collection_name="codebase",
            embedding_model="intfloat/multilingual-e5-small"
        )
    return _indexer

@mcp.tool()
def search_code(query: str, n_results: int = 10, diversity: float = 0.0, language: Optional[str] = None, chunk_type: Optional[str] = None) -> str:
    """Search code semantically using natural language.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return (1-50, default 10)
        diversity: Trade-off between relevance and diversity (0.0=pure relevance, 1.0=max diversity, default 0.0)
        language: Filter by programming language (e.g., 'python', 'javascript') 
        chunk_type: Filter by code chunk type (e.g., 'function', 'class', 'method')
        
    Returns:
        JSON string with search results
    """
    try:
        indexer = get_indexer()
        
        if language:
            results = indexer.search_by_language(query, [language], n_results)
        elif chunk_type:
            results = indexer.vector_store.search_by_chunk_type(query, [chunk_type], n_results)
        else:
            results = indexer.search_code(query, n_results, diversity=diversity)
        
        formatted_results = []
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            formatted_result = {
                "rank": i,
                "file_path": metadata['file_path'],
                "lines": f"{metadata['start_line']}-{metadata['end_line']}",
                "chunk_type": metadata['chunk_type'],
                "name": metadata.get('name', 'unnamed'),
                "language": metadata.get('language', 'unknown'),
                "similarity_score": 1 - result['distance'],
                "content": result['document'][:500] + "..." if len(result['document']) > 500 else result['document']
            }
            formatted_results.append(formatted_result)
        
        response = {
            "query": query,
            "total_results": len(results),
            "results": formatted_results
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return f"Search failed: {str(e)}"
if __name__ == "__main__":
    mcp.run()
