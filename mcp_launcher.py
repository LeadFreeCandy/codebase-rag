#!/usr/bin/env python3
"""Standalone MCP server launcher."""

import sys
import os
from pathlib import Path

# Add the rag directory to Python path
rag_dir = Path(__file__).parent
sys.path.insert(0, str(rag_dir))

# Now we can import and run the server
from codebase_rag.mcp_server import main

if __name__ == "__main__":
    main()