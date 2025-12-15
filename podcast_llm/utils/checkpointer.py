# podcast_llm/utils/checkpointer.py

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

The module uses JSON for serialization, providing a safe alternative to pickle
that avoids arbitrary code execution vulnerabilities while supporting the
specific data types used in podcast generation (Pydantic models, LangChain Documents).
"""


import json
import logging
from typing import Any, Callable, Union
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel

from podcast_llm.models import (
    PodcastOutline,
    Question,
    Answer,
)


logger = logging.getLogger(__name__)


# Type identifier constants for JSON serialization
_TYPE_KEY = '__type__'
_DATA_KEY = '__data__'
_TYPE_DOCUMENT = 'langchain_document'
_TYPE_PYDANTIC = 'pydantic_model'
_TYPE_LIST = 'typed_list'


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


# Registry of supported Pydantic model types for deserialization
_PYDANTIC_MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    'PodcastOutline': PodcastOutline,
    'Question': Question,
    'Answer': Answer,
}


def _serialize_value(value: Any) -> Any:
    """
    Serialize a value to a JSON-compatible format.
    
    Handles Pydantic models, LangChain Documents, and lists of these types.
    Simple types (str, int, float, bool, None, dict) are passed through.
    
    Args:
        value: The value to serialize
        
    Returns:
        A JSON-serializable representation of the value
        
    Raises:
        TypeError: If the value type is not supported for serialization
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    
    if isinstance(value, dict):
        # Check if it's already a simple dict (like final_script entries)
        # Recursively serialize dict values
        return {k: _serialize_value(v) for k, v in value.items()}
    
    if isinstance(value, Document):
        return {
            _TYPE_KEY: _TYPE_DOCUMENT,
            _DATA_KEY: {
                'page_content': value.page_content,
                'metadata': value.metadata,
            }
        }
    
    if isinstance(value, BaseModel):
        model_name = type(value).__name__
        if model_name not in _PYDANTIC_MODEL_REGISTRY:
            raise TypeError(
                f"Pydantic model type '{model_name}' is not registered for serialization. "
                f"Add it to _PYDANTIC_MODEL_REGISTRY in checkpointer.py"
            )
        return {
            _TYPE_KEY: _TYPE_PYDANTIC,
            'model_name': model_name,
            _DATA_KEY: value.model_dump(),
        }
    
    if isinstance(value, list):
        return {
            _TYPE_KEY: _TYPE_LIST,
            _DATA_KEY: [_serialize_value(item) for item in value],
        }
    
    raise TypeError(
        f"Cannot serialize value of type '{type(value).__name__}'. "
        f"Supported types: str, int, float, bool, None, dict, list, "
        f"Document, and registered Pydantic models."
    )


def _deserialize_value(data: Any) -> Any:
    """
    Deserialize a value from JSON format back to its original type.
    
    Reconstructs Pydantic models, LangChain Documents, and lists from
    their JSON representations.
    
    Args:
        data: The JSON-compatible data to deserialize
        
    Returns:
        The reconstructed Python object
        
    Raises:
        ValueError: If the data contains an unknown type identifier or model name
    """
    if data is None or isinstance(data, (str, int, float, bool)):
        return data
    
    if isinstance(data, dict):
        type_id = data.get(_TYPE_KEY)
        
        if type_id is None:
            # Regular dict - recursively deserialize values
            return {k: _deserialize_value(v) for k, v in data.items()}
        
        if type_id == _TYPE_DOCUMENT:
            doc_data = data[_DATA_KEY]
            return Document(
                page_content=doc_data['page_content'],
                metadata=doc_data.get('metadata', {}),
            )
        
        if type_id == _TYPE_PYDANTIC:
            model_name = data['model_name']
            model_class = _PYDANTIC_MODEL_REGISTRY.get(model_name)
            if model_class is None:
                raise ValueError(
                    f"Unknown Pydantic model type: '{model_name}'. "
                    f"Registered models: {list(_PYDANTIC_MODEL_REGISTRY.keys())}"
                )
            return model_class.model_validate(data[_DATA_KEY])
        
        if type_id == _TYPE_LIST:
            return [_deserialize_value(item) for item in data[_DATA_KEY]]
        
        raise ValueError(f"Unknown type identifier in checkpoint data: '{type_id}'")
    
    if isinstance(data, list):
        # Handle raw lists (shouldn't normally occur with our serialization)
        return [_deserialize_value(item) for item in data]
    
    return data


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
    - Uses JSON serialization to avoid security vulnerabilities associated with pickle
    
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
        """
        Execute a function with checkpointing support.
        
        If a checkpoint exists for the given stage, loads and returns the cached result.
        Otherwise, executes the function, saves the result to a checkpoint file, and
        returns the result.
        
        Args:
            fn: The function to execute if no checkpoint exists
            args: Arguments to pass to the function
            stage_name: Name of this processing stage (used in checkpoint filename)
            
        Returns:
            The result of the function (either from cache or fresh execution)
        """
        if not self.enabled:
            return fn(*args)

        # Generate checkpoint filename using base key
        checkpoint_file = self.checkpoint_dir / f'{self.checkpoint_key}_{stage_name}.json'

        # Try to load from checkpoint
        if checkpoint_file.exists():
            logger.info(f'Loading checkpoint from {checkpoint_file}')
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return _deserialize_value(data)
        
        # If it doesn't exist, call the function
        result = fn(*args)

        # Save checkpoint
        logger.info(f'Saving checkpoint to {checkpoint_file}')
        serialized = _serialize_value(result)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)

        return result
