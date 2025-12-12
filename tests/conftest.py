import os
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def set_test_env_vars():
    """Set up dummy environment variables for testing."""
    test_env_vars = {
        'GOOGLE_API_KEY': 'test-google-key',
        'ELEVENLABS_API_KEY': 'test-elevenlabs-key',
        'OPENAI_API_KEY': 'test-openai-key',
        'TAVILY_API_KEY': 'test-tavily-key',
        'ANTHROPIC_API_KEY': 'test-anthropic-key',
        'MOONSHOT_API_KEY': 'test-moonshot-key',
    }
    
    # Store original values
    original_values = {}
    for key in test_env_vars:
        original_values[key] = os.environ.get(key)
        os.environ[key] = test_env_vars[key]
    
    yield
    
    # Restore original values
    for key, value in original_values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def test_data_dir() -> Path:
    """Fixture that provides path to test data directory"""
    return Path(__file__).parent / 'test_data'
