"""
Utilities for checkpointing and resuming long-running processes.

This module provides functionality for saving and loading intermediate computation
results to disk, enabling efficient resumption of processing from the last successful
checkpoint. This is particularly useful for long-running podcast generation tasks
that may need to be interrupted and resumed.

Key components:
- Checkpointer: A class that manages saving/loading of checkpoint data with configurable
  paths and secure JSON serialization
- TypedSerializer: A secure serialization system that handles Pydantic models and 
  LangChain Documents
- to_snake_case: Helper function for converting checkpoint names to valid filenames

The checkpointing system helps with:
- Saving intermediate results during multi-step processing
- Resuming interrupted processes without recomputing completed steps  
- Debugging by examining saved checkpoint states
- Reducing wasted computation on process restarts

The module uses JSON-based serialization with type metadata to provide security,
debuggability, and cross-platform compatibility while avoiding the security risks
of pickle deserialization.
"""


import json
import logging
from typing import Any, Callable, Dict, List, Union
from pathlib import Path

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


class TypedSerializer:
    """
    A secure JSON-based serializer that handles complex Python objects with type safety.
    
    This serializer replaces pickle with a secure alternative that:
    - Uses JSON for human-readable, tamper-evident serialization
    - Includes type metadata for safe deserialization
    - Handles Pydantic models and LangChain Documents
    - Prevents arbitrary code execution during deserialization
    
    The serializer wraps objects in type-tagged envelopes that enable safe reconstruction
    without executing untrusted code.
    """
    
    @staticmethod
    def serialize(obj: Any) -> Dict[str, Any]:
        """
        Serialize an object to a JSON-compatible dictionary with type metadata.
        
        Args:
            obj: The object to serialize
            
        Returns:
            Dict containing the serialized object and type metadata
        """
        # Handle None
        if obj is None:
            return {"__type__": "NoneType", "data": None}
            
        # Handle basic JSON types
        if isinstance(obj, (str, int, float, bool)):
            return {"__type__": type(obj).__name__, "data": obj}
            
        # Handle lists
        if isinstance(obj, list):
            return {
                "__type__": "list",
                "data": [TypedSerializer.serialize(item) for item in obj]
            }
            
        # Handle dictionaries
        if isinstance(obj, dict):
            return {
                "__type__": "dict", 
                "data": {key: TypedSerializer.serialize(value) for key, value in obj.items()}
            }
            
        # Handle Pydantic models
        if isinstance(obj, BaseModel):
            return {
                "__type__": "pydantic_model",
                "__class__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
                "data": obj.model_dump()
            }
            
        # Handle LangChain Document objects
        if hasattr(obj, 'page_content') and hasattr(obj, 'metadata'):
            return {
                "__type__": "langchain_document",
                "data": {
                    "page_content": obj.page_content,
                    "metadata": obj.metadata
                }
            }
            
        # Fallback for unknown types - convert to string representation
        logger.warning(f"Serializing unknown type {type(obj)} as string representation")
        return {
            "__type__": "unknown",
            "__original_type__": str(type(obj)),
            "data": str(obj)
        }
    
    @staticmethod  
    def deserialize(data: Dict[str, Any]) -> Any:
        """
        Deserialize a type-tagged dictionary back to the original object.
        
        Args:
            data: Dictionary containing serialized object and type metadata
            
        Returns:
            The reconstructed object
        """
        if not isinstance(data, dict) or "__type__" not in data:
            return data
            
        obj_type = data["__type__"]
        obj_data = data["data"]
        
        # Handle None
        if obj_type == "NoneType":
            return None
            
        # Handle basic types
        if obj_type in ("str", "int", "float", "bool"):
            return obj_data
            
        # Handle lists
        if obj_type == "list":
            return [TypedSerializer.deserialize(item) for item in obj_data]
            
        # Handle dictionaries
        if obj_type == "dict":
            return {key: TypedSerializer.deserialize(value) for key, value in obj_data.items()}
            
        # Handle Pydantic models
        if obj_type == "pydantic_model":
            class_path = data["__class__"]
            module_name, class_name = class_path.rsplit(".", 1)
            
            # Import the module and get the class
            import importlib
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls.model_validate(obj_data)
            except (ImportError, AttributeError) as e:
                logger.error(f"Failed to deserialize Pydantic model {class_path}: {e}")
                return obj_data
                
        # Handle LangChain Documents
        if obj_type == "langchain_document":
            try:
                from langchain_core.documents import Document
                return Document(
                    page_content=obj_data["page_content"],
                    metadata=obj_data["metadata"]
                )
            except ImportError as e:
                logger.error(f"Failed to import LangChain Document: {e}")
                return obj_data
                
        # Handle unknown types
        if obj_type == "unknown":
            logger.warning(f"Deserializing unknown type {data.get('__original_type__')} as string")
            return obj_data
            
        # Fallback
        logger.warning(f"Unknown serialization type: {obj_type}")
        return obj_data


class Checkpointer:
    """
    A class for managing checkpointing of intermediate results during processing.

    The Checkpointer allows saving and loading of intermediate computation results to disk,
    enabling resumption of long-running processes from the last successful checkpoint.
    
    This implementation uses secure JSON-based serialization instead of pickle to avoid
    code execution vulnerabilities while maintaining full functionality.
    
    Key features:
    - Configurable checkpoint directory and key prefix for files
    - Can be enabled/disabled via constructor
    - Automatically creates checkpoint directory if needed
    - Saves results as JSON files with type metadata for security
    - Loads from existing checkpoints when available
    - Backwards compatible with existing pickle files during migration
    
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

    def _get_checkpoint_file(self, stage_name: str, format_type: str = 'json') -> Path:
        """Get the checkpoint file path for a given stage and format."""
        extension = '.json' if format_type == 'json' else '.pkl'
        return self.checkpoint_dir / f'{self.checkpoint_key}_{stage_name}{extension}'

    def _load_from_pickle(self, checkpoint_file: Path) -> Any:
        """Load from legacy pickle file with warning."""
        logger.warning(f"Loading from legacy pickle file: {checkpoint_file}")
        logger.warning("Consider regenerating checkpoints to use secure JSON format")
        
        import pickle
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)

    def _load_from_json(self, checkpoint_file: Path) -> Any:
        """Load from secure JSON file."""
        logger.info(f'Loading checkpoint from {checkpoint_file}')
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return TypedSerializer.deserialize(data)

    def _save_to_json(self, result: Any, checkpoint_file: Path) -> None:
        """Save to secure JSON file."""
        logger.info(f'Saving checkpoint to {checkpoint_file}')
        serialized_data = TypedSerializer.serialize(result)
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)

    def checkpoint(self, fn: Callable, args: list, stage_name: str = 'result') -> Any:
        """
        Execute a function and checkpoint its result, or load from existing checkpoint.
        
        Args:
            fn: Function to execute if no checkpoint exists
            args: Arguments to pass to the function
            stage_name: Name for this checkpoint stage
            
        Returns:
            The function result, either computed or loaded from checkpoint
        """
        if not self.enabled:
            return fn(*args)

        # Check for existing JSON checkpoint first
        json_checkpoint_file = self._get_checkpoint_file(stage_name, 'json')
        if json_checkpoint_file.exists():
            return self._load_from_json(json_checkpoint_file)
            
        # Check for legacy pickle checkpoint
        pickle_checkpoint_file = self._get_checkpoint_file(stage_name, 'pkl')
        if pickle_checkpoint_file.exists():
            result = self._load_from_pickle(pickle_checkpoint_file)
            # Migrate to JSON format
            logger.info(f"Migrating checkpoint to JSON format: {json_checkpoint_file}")
            self._save_to_json(result, json_checkpoint_file)
            return result
        
        # No checkpoint exists, execute function
        result = fn(*args)

        # Save checkpoint in JSON format
        self._save_to_json(result, json_checkpoint_file)

        return result