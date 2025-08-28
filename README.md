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
