"""Command-line interface for codebase RAG system."""

import json
import sys
from pathlib import Path
from typing import Optional

import click

from .indexer import CodebaseIndexer, IncrementalIndexer
from .config import ConfigManager, RAGConfig


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    ctx.ensure_object(dict)
    
    config_manager = ConfigManager(config)
    ctx.obj['config_manager'] = config_manager
    ctx.obj['config'] = config_manager.get_config()


@cli.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--clear', is_flag=True, help='Clear existing index before indexing')
@click.option('--no-cache', is_flag=True, help='Disable chunk caching')
@click.option('--clear-cache', is_flag=True, help='Clear chunk cache before indexing')
@click.option('--persist-dir', help='Directory to store the vector database')
@click.option('--collection', help='Name of the collection')
@click.option('--model', help='Embedding model to use')
@click.pass_context
def index(ctx, codebase_path, clear, no_cache, clear_cache, persist_dir, collection, model):
    """Index a codebase for semantic search."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    if model:
        config.embedding_model = model
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            ignore_patterns=config.ignore_patterns
        )
        
        click.echo(f"Indexing codebase: {codebase_path}")
        if clear:
            click.echo("Clearing existing index...")
        if clear_cache:
            click.echo("Clearing chunk cache...")
            indexer.clear_chunk_cache(codebase_path)
        
        use_cache = not no_cache
        if no_cache:
            click.echo("Chunk caching disabled")
        
        stats = indexer.index_codebase(codebase_path, clear_existing=clear, use_cache=use_cache)
        
        click.echo(f"\nIndexing completed successfully!")
        click.echo(f"Database saved to: {config.persist_directory}")
        
    except Exception as e:
        click.echo(f"Error during indexing: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('query')
@click.option('--n-results', '-n', default=10, help='Number of results to return')
@click.option('--diversity', '-d', default=0.0, type=float, help='Diversity factor (0.0=pure relevance, 1.0=max diversity)')
@click.option('--language', '-l', help='Filter by programming language')
@click.option('--chunk-type', '-t', help='Filter by chunk type (function, class, method)')
@click.option('--persist-dir', help='Directory where the vector database is stored')
@click.option('--collection', help='Name of the collection')
@click.pass_context
def search(ctx, query, n_results, diversity, language, chunk_type, persist_dir, collection):
    """Search the indexed codebase."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
        
        if language:
            results = indexer.search_by_language(query, [language], n_results)
        elif chunk_type:
            results = indexer.vector_store.search_by_chunk_type(query, [chunk_type], n_results)
        else:
            results = indexer.search_code(query, n_results, diversity=diversity)
        
        if not results:
            click.echo("No results found.")
            return
        
        click.echo(f"\nFound {len(results)} results for: '{query}'\n")
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity = 1 - result['distance']
            
            click.echo(f"Result {i} (similarity: {similarity:.3f})")
            click.echo(f"File: {metadata['file_path']}")
            click.echo(f"Lines: {metadata['start_line']}-{metadata['end_line']}")
            click.echo(f"Type: {metadata['chunk_type']}")
            if 'name' in metadata:
                click.echo(f"Name: {metadata['name']}")
            if 'language' in metadata:
                click.echo(f"Language: {metadata['language']}")
            
            content = result['document']
            if len(content) > 300:
                content = content[:300] + "..."
            
            click.echo(f"Content:\n{content}")
            click.echo("-" * 80)
        
    except Exception as e:
        click.echo(f"Error during search: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--persist-dir', help='Directory where the vector database is stored')
@click.option('--collection', help='Name of the collection')
@click.pass_context
def info(ctx, persist_dir, collection):
    """Get information about the indexed codebase."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
        
        info = indexer.get_collection_info()
        coverage = indexer.get_file_coverage()
        
        click.echo("Collection Information:")
        click.echo(f"  Name: {info['name']}")
        click.echo(f"  Total chunks: {info['count']}")
        click.echo(f"  Database location: {info['persist_directory']}")
        
        if 'last_indexing_stats' in info:
            stats = info['last_indexing_stats']
            click.echo(f"  Files indexed: {stats['total_files']}")
            click.echo(f"  Processing time: {stats['processing_time']}")
        
        click.echo(f"\nFile Coverage (top 10):")
        sorted_files = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        for file_path, chunk_count in sorted_files[:10]:
            click.echo(f"  {file_path}: {chunk_count} chunks")
        
        if len(sorted_files) > 10:
            click.echo(f"  ... and {len(sorted_files) - 10} more files")
        
    except Exception as e:
        click.echo(f"Error getting collection info: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--persist-dir', help='Directory where the vector database is stored')
@click.option('--collection', help='Name of the collection')
@click.pass_context
def update(ctx, codebase_path, persist_dir, collection):
    """Incrementally update the index for changed files."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model,
            ignore_patterns=config.ignore_patterns
        )
        
        incremental_indexer = IncrementalIndexer(indexer)
        incremental_indexer.incremental_update(codebase_path)
        
        click.echo("Incremental update completed successfully!")
        
    except Exception as e:
        click.echo(f"Error during update: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('output_path')
@click.option('--persist-dir', help='Directory where the vector database is stored')
@click.option('--collection', help='Name of the collection')
@click.pass_context
def export(ctx, output_path, persist_dir, collection):
    """Export the vector index to a JSON file."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
        
        indexer.export_index(output_path)
        click.echo(f"Index exported to: {output_path}")
        
    except Exception as e:
        click.echo(f"Error during export: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config-file', help='Configuration file for the MCP server')
@click.pass_context
def server(ctx, config_file):
    """Start the MCP server."""
    import asyncio
    
    try:
        server = CodebaseRAGServer(config_file)
        click.echo("Starting MCP server...")
        click.echo("Use Ctrl+C to stop the server")
        asyncio.run(server.run_server())
        
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--persist-dir', help='Directory where the vector database is stored')
@click.option('--collection', help='Name of the collection')
@click.pass_context
def clear_cache(ctx, codebase_path, persist_dir, collection):
    """Clear chunk cache for a codebase."""
    config = ctx.obj['config']
    
    if persist_dir:
        config.persist_directory = persist_dir
    if collection:
        config.collection_name = collection
    
    try:
        indexer = CodebaseIndexer(
            persist_directory=config.persist_directory,
            collection_name=config.collection_name,
            embedding_model=config.embedding_model
        )
        
        success = indexer.clear_chunk_cache(codebase_path)
        if success:
            click.echo("Chunk cache cleared successfully!")
        else:
            click.echo("No chunk cache found to clear.")
        
    except Exception as e:
        click.echo(f"Error clearing cache: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def config(ctx):
    """Configuration management commands."""
    pass


@config.command('show')
@click.pass_context
def config_show(ctx):
    """Show current configuration."""
    config_manager = ctx.obj['config_manager']
    config_manager.print_config()


@config.command('create-sample')
@click.argument('output_path', default='rag_config.json')
@click.pass_context
def config_create_sample(ctx, output_path):
    """Create a sample configuration file."""
    config_manager = ctx.obj['config_manager']
    config_manager.create_sample_config(output_path)


@config.command('validate')
@click.argument('config_path', type=click.Path(exists=True))
def config_validate(config_path):
    """Validate a configuration file."""
    try:
        config_manager = ConfigManager(config_path)
        click.echo(f"Configuration file is valid: {config_path}")
        config_manager.print_config()
        
    except Exception as e:
        click.echo(f"Configuration file is invalid: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()