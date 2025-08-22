#!/usr/bin/env python3
"""Entry point script for the Codebase RAG MCP Server."""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the MCP server
if __name__ == "__main__":
    from mcp_server_fixed import mcp
    mcp.run()