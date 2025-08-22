"""Configuration management for codebase RAG system."""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml


@dataclass
class RAGConfig:
    persist_directory: str = "./chroma_db"
    collection_name: str = "codebase"
    # embedding_model: str = "all-MiniLM-L6-v2"
    # embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    # embedding_model: str = "Snowflake/snowflake-arctic-embed-xs"
    embedding_model: str = "intfloat/multilingual-e5-small"
    # embedding_model: str = "Snowflake/snowflake-arctic-embed-l-v2.0"
    max_results: int = 20
    ignore_patterns: List[str] = None
    batch_size: int = 100
    max_file_size_mb: int = 10
    chunk_size: int = 8000
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = [
                ".git/", "__pycache__/", "node_modules/", ".venv/", "venv/", ".env/",
                "*.pyc", "*.pyo", "*.so", "*.dll", "*.exe",
                "*.jpg", "*.jpeg", "*.png", "*.gif", "*.pdf", "*.zip", "*.tar.gz",
                "*.log", "*.tmp", "*.cache", "*.bak", "*.swp",
                ".DS_Store", "Thumbs.db"
            ]


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = RAGConfig()
        
        if config_path:
            self.load_config(config_path)
        else:
            self._try_load_default_configs()
    
    def _try_load_default_configs(self):
        default_paths = [
            "rag_config.json",
            "rag_config.yaml",
            "config.json", 
            "config.yaml",
            os.path.expanduser("~/.config/codebase_rag/config.json"),
            os.path.expanduser("~/.config/codebase_rag/config.yaml"),
        ]
        
        for path in default_paths:
            if Path(path).exists():
                self.load_config(path)
                self.config_path = path
                break
    
    def load_config(self, config_path: str):
        config_file = Path(config_path)
        
        if not config_file.exists():
            print(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            self._update_config_from_dict(data)
            print(f"Loaded configuration from: {config_path}")
            
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}")
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' ignored")
    
    def save_config(self, config_path: str):
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self.config)
        
        try:
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            print(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            print(f"Error saving config file {config_path}: {e}")
    
    def create_sample_config(self, output_path: str = "rag_config.json"):
        sample_config = RAGConfig()
        
        with open(output_path, 'w') as f:
            json.dump(asdict(sample_config), f, indent=2)
        
        print(f"Sample configuration created: {output_path}")
    
    def get_config(self) -> RAGConfig:
        return self.config
    
    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config key '{key}' ignored")
    
    def print_config(self):
        config_dict = asdict(self.config)
        print("Current Configuration:")
        print(json.dumps(config_dict, indent=2))


def load_config_from_env() -> RAGConfig:
    config = RAGConfig()
    
    env_mappings = {
        'CODEBASE_RAG_PERSIST_DIR': 'persist_directory',
        'CODEBASE_RAG_COLLECTION': 'collection_name',
        'CODEBASE_RAG_EMBEDDING_MODEL': 'embedding_model',
        'CODEBASE_RAG_MAX_RESULTS': 'max_results',
        'CODEBASE_RAG_BATCH_SIZE': 'batch_size',
        'CODEBASE_RAG_MAX_FILE_SIZE_MB': 'max_file_size_mb',
        'CODEBASE_RAG_CHUNK_SIZE': 'chunk_size'
    }
    
    for env_var, config_attr in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value is not None:
            if config_attr in ['max_results', 'batch_size', 'max_file_size_mb', 'chunk_size']:
                try:
                    env_value = int(env_value)
                except ValueError:
                    print(f"Warning: Invalid integer value for {env_var}: {env_value}")
                    continue
            
            setattr(config, config_attr, env_value)
    
    ignore_patterns_env = os.getenv('CODEBASE_RAG_IGNORE_PATTERNS')
    if ignore_patterns_env:
        try:
            config.ignore_patterns = json.loads(ignore_patterns_env)
        except json.JSONDecodeError:
            config.ignore_patterns = [pattern.strip() for pattern in ignore_patterns_env.split(',')]
    
    return config
