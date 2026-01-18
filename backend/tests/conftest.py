import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from typing import Generator
import os
import sys
import types

# Set test environment variables before importing app
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_USER", "test_user")
os.environ.setdefault("POSTGRES_PASSWORD", "test_password")
os.environ.setdefault("POSTGRES_DB", "test_db")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")

# Mock heavy dependencies in sys.modules before any imports
# This prevents actual imports of whisper, spacy, etc.
mock_whisper = types.ModuleType('whisper')
mock_whisper.load_model = MagicMock()
sys.modules['whisper'] = mock_whisper

mock_spacy = types.ModuleType('spacy')
mock_spacy.load = MagicMock(return_value=MagicMock())
sys.modules['spacy'] = mock_spacy

mock_tts = types.ModuleType('TTS')
mock_tts.api = types.ModuleType('TTS.api')
sys.modules['TTS'] = mock_tts
sys.modules['TTS.api'] = mock_tts.api

# Mock sentence_transformers
mock_st = types.ModuleType('sentence_transformers')
mock_st.SentenceTransformer = MagicMock
sys.modules['sentence_transformers'] = mock_st

from main import app
from database import pool, init_db, close_pool
from redis_client import redis_client, init_redis


@pytest.fixture(scope="session")
def mock_db_pool():
    """Mock database connection pool"""
    with patch('database.SimpleConnectionPool') as mock_pool_class:
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        # Mock connection context manager
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        mock_pool.getconn.return_value = mock_conn
        mock_pool.putconn.return_value = None
        
        yield mock_pool, mock_conn, mock_cur


@pytest.fixture(scope="session")
def mock_redis():
    """Mock Redis client"""
    with patch('redis_client.redis.Redis') as mock_redis_class:
        mock_redis_instance = MagicMock()
        mock_redis_class.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.get.return_value = None
        mock_redis_instance.setex.return_value = True
        mock_redis_instance.delete.return_value = True
        yield mock_redis_instance


@pytest.fixture(scope="function")
def client(mock_db_pool, mock_redis):
    """Create test client with mocked dependencies"""
    with patch('main.init_db'), patch('main.init_redis'):
        with patch('database.get_connection') as mock_get_conn:
            mock_pool, mock_conn, mock_cur = mock_db_pool
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            mock_get_conn.return_value.__exit__.return_value = None
            
            with patch('redis_client.redis_client', mock_redis):
                with TestClient(app) as test_client:
                    yield test_client


@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "document_id": "test-doc-123",
        "filename": "test.pdf",
        "file_type": "pdf",
        "status": "completed",
        "chunks": 5,
        "entities": 10
    }


@pytest.fixture
def sample_query_request():
    """Sample query request"""
    return {
        "query": "What is Rwanda?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response"""
    return {
        "query": "What is Rwanda?",
        "rectified_query": "What is Rwanda?",
        "response": "Rwanda is a country in East Africa.",
        "source": "vector_db",
        "entities": [{"text": "Rwanda", "label": "GPE"}],
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_audio_data():
    """Sample audio data (mock)"""
    return b"fake audio data"


@pytest.fixture
def mock_embedding():
    """Mock embedding vector"""
    return [0.1] * 384  # all-MiniLM-L6-v2 produces 384-dimensional embeddings


@pytest.fixture
def mock_llm_response():
    """Mock LLM response"""
    mock_response = MagicMock()
    mock_response.content = "This is a test response from the LLM."
    return mock_response
