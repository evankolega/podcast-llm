"""
Utilities for checkpointing and resuming long-running processes.

This module provides functionality for saving and loading intermediate computation
results to disk, enabling efficient resumption of processing from the last successful
checkpoint. This is particularly useful for long-running podcast generation tasks
that may need to be interrupted and resumed.

Key components:
- Checkpointer: A class that manages saving/loading of checkpoint data with configurable
  paths and serialization
- to_snake_case: Helper function for converting checkpoint names to valid filenames

The checkpointing system helps with:
- Saving intermediate results during multi-step processing
- Resuming interrupted processes without recomputing completed steps  
- Debugging by examining saved checkpoint states
- Reducing wasted computation on process restarts

The module uses JSON for serialization to ensure security and avoid arbitrary code
execution vulnerabilities that can occur with pickle deserialization.
"""


import json
import logging
from typing import Any, Callable
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel


logger = logging.getLogger(__name__)


def to_snake_case(text: str) -> str:
    """
    Convert a string to snake_case format.
    
    Takes any string input and converts it to snake_case by:
    1. Replacing spaces and hyphens with underscores
    2. Converting to lowercase
    3. Removing any non-alphanumeric characters except underscores
    
    Args:
        text (str): Input string to convert
        
    Returns:
        str: Snake case formatted string
    """
    # Replace spaces and hyphens with underscores
    text = text.replace(' ', '_').replace('-', '_')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove any characters that aren't alphanumeric or underscore
    text = ''.join(c for c in text if c.isalnum() or c == '_')
    
    # Replace multiple consecutive underscores with single underscore
    while '__' in text:
        text = text.replace('__', '_')
        
    # Remove leading/trailing underscores
    return text.strip('_')


def _serialize_value(value: Any) -> Any:
    """
    Recursively serialize a value to a JSON-compatible format.
    
    Handles:
    - Pydantic BaseModel instances (converted via model_dump with type marker)
    - LangChain Document instances (converted to dict with type marker)
    - Lists (recursively serialized)
    - Dicts (recursively serialized)
    - Primitive types (returned as-is)
    
    Args:
        value: The value to serialize
        
    Returns:
        A JSON-serializable representation of the value
    """
    if isinstance(value, BaseModel):
        return {
            '__type__': f'pydantic:{value.__class__.__module__}.{value.__class__.__name__}',
            'data': value.model_dump()
        }
    elif isinstance(value, Document):
        return {
            '__type__': 'langchain_document',
            'page_content': value.page_content,
            'metadata': value.metadata
        }
    elif isinstance(value, list):
        return [_serialize_value(item) for item in value]
    elif isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    else:
        return value


def _deserialize_value(value: Any) -> Any:
    """
    Recursively deserialize a value from JSON format back to Python objects.
    
    Handles:
    - Pydantic models (reconstructed from type marker and data)
    - LangChain Document instances (reconstructed from dict representation)
    - Lists (recursively deserialized)
    - Dicts (recursively deserialized, unless they have type markers)
    - Primitive types (returned as-is)
    
    Args:
        value: The JSON-deserialized value to convert back to Python objects
        
    Returns:
        The reconstructed Python object
        
    Raises:
        ValueError: If a Pydantic model type cannot be found
    """
    if isinstance(value, dict):
        if '__type__' in value:
            type_marker = value['__type__']
            
            if type_marker == 'langchain_document':
                return Document(
                    page_content=value['page_content'],
                    metadata=value.get('metadata', {})
                )
            elif type_marker.startswith('pydantic:'):
                # Extract module and class name
                full_class_path = type_marker[len('pydantic:'):]
                module_path, class_name = full_class_path.rsplit('.', 1)
                
                # Import the module and get the class
                import importlib
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
                
                return model_class.model_validate(value['data'])
            else:
                logger.warning(f"Unknown type marker '{type_marker}', returning as dict")
                return {k: _deserialize_value(v) for k, v in value.items() if k != '__type__'}
        else:
            return {k: _deserialize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    else:
        return value


class Checkpointer:
    """
    A class for managing checkpointing of intermediate results during processing.

    The Checkpointer allows saving and loading of intermediate computation results to disk,
    enabling resumption of long-running processes from the last successful checkpoint.
    
    Key features:
    - Configurable checkpoint directory and key prefix for files
    - Can be enabled/disabled via constructor
    - Automatically creates checkpoint directory if needed
    - Saves results as JSON files with stage-specific names
    - Loads from existing checkpoints when available
    - Supports Pydantic models, LangChain Documents, and plain Python types
    
    Example usage:
        checkpointer = Checkpointer(
            checkpoint_key='my_process_',
            enabled=True
        )
        
        # Will save result to disk and return it
        result = checkpointer.checkpoint(
            expensive_computation(), 
            stage_name='stage1'
        )
        
        # On subsequent runs, will load from disk instead of recomputing
        result = checkpointer.checkpoint(
            expensive_computation(),
            stage_name='stage1'
        )
    """
    def __init__(self, checkpoint_key: str, checkpoint_dir: str = '.checkpoints', enabled: bool = True):
        """
        Initialize the Checkpointer.

        Args:
            checkpoint_key (str): Base key to use for checkpoint filenames
            checkpoint_dir (str): Directory path for storing checkpoints
            enabled (bool): Whether to enable checkpointing functionality
        """
        logger.info(f"Initializing checkpointer with key: {checkpoint_key}")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.enabled = enabled
        self.checkpoint_key = checkpoint_key
        if enabled:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, fn: Callable, args: list, stage_name: str = 'result') -> Any:
        if not self.enabled:
            return fn(*args)

        # Generate checkpoint filename using base key (now with .json extension)
        checkpoint_file = self.checkpoint_dir / f'{self.checkpoint_key}_{stage_name}.json'

        # Try to load from checkpoint
        if checkpoint_file.exists():
            logger.info(f'Loading checkpoint from {checkpoint_file}')
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                serialized_data = json.load(f)
                return _deserialize_value(serialized_data)
        
        # If it doesn't exist, call the function
        result = fn(*args)

        # Save checkpoint
        logger.info(f'Saving checkpoint to {checkpoint_file}')
        serialized_data = _serialize_value(result)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)

        return result
