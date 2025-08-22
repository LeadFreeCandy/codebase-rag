# Codebase RAG: AI-Powered Code Search and Analysis

A powerful RAG (Retrieval Augmented Generation) system for intelligent codebase search and analysis. Built with state-of-the-art embeddings and tree-sitter parsing for accurate code understanding.

## Installation

### Prerequisites

- Python 3.8 or higher
- Git

### Quick Install

1. **Clone the repository:**
   ```bash
   git clone https://github.com/appfolio/codebase-rag
   cd codebase-rag
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

## Building Your Index

### Command Line Interface

**Index a codebase:**
```bash
codebase-rag index /path/to/your/codebase --persist-dir ./apm_bundle_full_db
```

**Index a codebase from scratch:**
```bash
codebase-rag index /path/to/your/codebase --clear --clear-cache --persist-dir ./apm_bundle_full_db
```

**Search your codebase:**
```bash
codebase-rag search "function that calculates totals" --n-results 5 --persist-dir ./apm_bundle_full_db
```

**Advanced search options:**
```bash
# Search with diversity (reduces redundant results)
codebase-rag search "authentication logic" --diversity 0.5 --persist-dir ./apm_bundle_full_db

# Search specific languages
codebase-rag search "error handling" --language python --language javascript --persist-dir ./apm_bundle_full_db

# Search specific chunk types
codebase-rag search "class definition" --chunk-type class --persist-dir ./apm_bundle_full_db

# Example with your specific query format
codebase-rag search "how do you undo a receivable payment?" --persist-dir ./apm_bundle_full_db
```

## MCP Server Setup

### Configuration for Claude Code/VS Code

The MCP server allows integration with Claude Code and other MCP-compatible tools. Here are the configuration examples:

**For Claude Code:**
Edit `~/.anthropic/claude_desktop_config.json`:

```json
{
  "servers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "/path/to/codebase-rag/venv/bin/python",
      "args": ["/path/to/codebase-rag/mcp_server_fixed.py"],
      "cwd": "/path/to/codebase-rag"
    }
  },
  "inputs": []
}
```

**Example with specific path (replace with your actual path):**
```json
{
  "servers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "/Users/samir.beall/src/rag/venv/bin/python",
      "args": ["/Users/samir.beall/src/rag/mcp_server_fixed.py"],
      "cwd": "/Users/samir.beall/src/rag"
    }
  },
  "inputs": []
}
```

**For VS Code with MCP Extension:**
Edit your MCP extension settings:

```json
{
  "servers": {
    "codebase-rag": {
      "type": "stdio",
      "command": "/path/to/codebase-rag/venv/bin/python",
      "args": ["/path/to/codebase-rag/mcp_server_fixed.py"],
      "cwd": "/path/to/codebase-rag"
    }
  },
  "inputs": []
}
```

### Starting the Server Manually

```bash
# Start the MCP server
python mcp_server_fixed.py

# Or use the CLI command (if installed with pip)
python -m codebase_rag.cli server
```

## Configuration

### Embedding Models

The system supports various embedding models. Recommended options:

**Fast & Efficient:**
- `intfloat/multilingual-e5-small` (384 dim, ~60 chunks/s) - **Recommended**
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, ~160 chunks/s)

**High Quality:**
- `sentence-transformers/all-mpnet-base-v2` (768 dim, ~28 chunks/s)
- `Snowflake/snowflake-arctic-embed-xs` (384 dim, ~100 chunks/s)

### Custom Configuration

Create a `config.yaml` file:

```yaml
# Model settings
embedding_model: "intfloat/multilingual-e5-small"
device: "mps"  # or "cuda", "cpu"

# Database settings
persist_directory: "./codebase_db"
collection_name: "my_codebase"

# Chunking settings
max_chunk_size: 8000
min_chunk_size: 25
batch_size: 16

# File filters
ignore_patterns:
  - "*.log"
  - "node_modules/**"
  - ".git/**"
  - "**/__pycache__/**"

# Language support
supported_languages:
  - python
  - javascript
  - typescript
  - go
  - rust
  - java
  - cpp
  - ruby
```

Use with CLI:
```bash
python -m codebase_rag.cli --config config.yaml index /path/to/codebase
```

## Usage Examples

### Basic Search
```bash
# Find authentication-related code
python -m codebase_rag.cli search "user authentication and login" --persist-dir ./apm_bundle_full_db

# Find error handling patterns
python -m codebase_rag.cli search "exception handling and error management" --persist-dir ./apm_bundle_full_db

# Find specific functions
python -m codebase_rag.cli search "function that validates email addresses" --persist-dir ./apm_bundle_full_db
```

### Advanced Search
```bash
# Search with diversity to get varied results
python -m codebase_rag.cli search "database operations" --diversity 0.3 --persist-dir ./apm_bundle_full_db

# Search only Python files
python -m codebase_rag.cli search "async functions" --language python --persist-dir ./apm_bundle_full_db

# Search only class definitions
python -m codebase_rag.cli search "user management" --chunk-type class --persist-dir ./apm_bundle_full_db

# Get more results
python -m codebase_rag.cli search "API endpoints" --n-results 20 --persist-dir ./apm_bundle_full_db
```

### MCP Integration

Once configured, you can use these commands in Claude Code:

- **Search your codebase:** "Search for functions that handle file uploads"
- **Get collection info:** "What's the current status of my codebase index?"
- **Targeted searches:** "Find authentication logic in Python files"

## Performance

### Typical Performance Metrics

| Model | Speed (chunks/s) | Dimension | Memory Usage | Quality |
|-------|------------------|-----------|--------------|---------|
| E5-Small | 60 | 384 | Low | High |
| MiniLM-L6 | 160 | 384 | Low | Good |
| MiniLM-L12 | 195 | 384 | Medium | High |
| MPNet-Base | 28 | 768 | High | Excellent |

### Optimization Tips

1. **Use SSD storage** for the database directory
2. **Enable GPU** if available (CUDA or MPS)
3. **Increase batch size** for larger codebases
4. **Use E5-Small model** for best balance of speed and quality
5. **Filter unnecessary files** with ignore patterns

## Troubleshooting

### Common Issues

**"No chunks found to index"**
- Ensure the path is a git repository or contains code files
- Check ignore patterns aren't too restrictive

**"Model loading failed"**
- Verify internet connection for model download
- Check available disk space (~2GB for models)
- Try CPU device if GPU issues occur

**"MCP server not connecting"**
- Verify Python path in MCP configuration
- Check that all dependencies are installed
- Ensure the database directory exists and has the pre-built index

**Slow indexing performance**
- Try a faster model like MiniLM-L6
- Reduce batch size if running out of memory
- Use CPU device if MPS/CUDA issues occur

### Debug Mode

Enable debug logging:
```bash
export CODEBASE_RAG_DEBUG=1
python -m codebase_rag.cli search "your query" --persist-dir ./apm_bundle_full_db
```

### Getting Help

1. Check the error logs in `./logs/codebase_rag.log`
2. Verify your configuration with `python -m codebase_rag.cli info --persist-dir ./apm_bundle_full_db`
3. Test the MCP server manually: `python mcp_server_fixed.py`

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Tree-sitter   │───▶│   Code Chunker   │───▶│   E5 Embedder   │
│   Parser        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MCP Server    │◀───│   Vector Store   │◀───│   ChromaDB      │
│                 │    │   (RAG Engine)   │    │   with FAISS    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Files Included

This distribution includes:

- **`codebase_rag/`** - Main Python package with all RAG functionality
- **`apm_bundle_full_db/`** - Pre-built index database (ready to use)
- **`mcp_server_fixed.py`** - MCP server implementation
- **`run_mcp_server.py`** - Server entry point script
- **`requirements.txt`** - All Python dependencies
- **`setup.py`** - Package installation script

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the search (using pre-built index):**
   ```bash
   python -m codebase_rag.cli search "authentication function" --n-results 3 --persist-dir ./apm_bundle_full_db
   ```

3. **Start MCP server:**
   ```bash
   python mcp_server_fixed.py
   ```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Built with [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- Uses [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) for code parsing
- Powered by [E5 embeddings](https://huggingface.co/intfloat/multilingual-e5-small) for semantic understanding
- MCP integration via [Model Context Protocol](https://modelcontextprotocol.io/)
