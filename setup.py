#!/usr/bin/env python3
"""Setup script for Codebase RAG."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="codebase-rag",
    version="1.0.0",
    description="AI-powered codebase search and analysis using RAG (Retrieval Augmented Generation)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Codebase RAG",
    author_email="codebase-rag@example.com",
    url="https://github.com/yourusername/codebase-rag",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "codebase-rag=codebase_rag.cli:main",
            "codebase-rag-server=run_mcp_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries",
        "Topic :: Text Processing :: Indexing",
    ],
    keywords="rag, codebase, search, ai, embeddings, mcp",
    include_package_data=True,
    zip_safe=False,
)