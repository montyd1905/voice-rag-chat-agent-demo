import pytest
from models import QueryRequest, QueryResponse, DocumentStatusResponse
from pydantic import ValidationError


class TestQueryRequest:
    """Test QueryRequest model"""
    
    def test_query_request_valid(self):
        """Test valid query request"""
        request = QueryRequest(query="What is Rwanda?", session_id="test-session")
        assert request.query == "What is Rwanda?"
        assert request.session_id == "test-session"
    
    def test_query_request_without_session_id(self):
        """Test query request without session ID"""
        request = QueryRequest(query="What is Rwanda?")
        assert request.query == "What is Rwanda?"
        assert request.session_id is None
    
    def test_query_request_empty_query(self):
        """Test query request with empty query (Pydantic v2 allows empty strings)"""
        # Pydantic v2 allows empty strings by default, so this should not raise
        request = QueryRequest(query="")
        assert request.query == ""


class TestQueryResponse:
    """Test QueryResponse model"""
    
    def test_query_response_valid(self):
        """Test valid query response"""
        response = QueryResponse(
            query="What is Rwanda?",
            rectified_query="What is Rwanda?",
            response="Rwanda is a country.",
            source="vector_db",
            entities=[{"text": "Rwanda", "label": "GPE"}],
            session_id="test-session"
        )
        assert response.query == "What is Rwanda?"
        assert response.response == "Rwanda is a country."
        assert response.source == "vector_db"
        assert len(response.entities) == 1
    
    def test_query_response_without_rectified_query(self):
        """Test query response without rectified query"""
        response = QueryResponse(
            query="What is Rwanda?",
            response="Rwanda is a country.",
            source="vector_db",
            entities=[],
            session_id="test-session"
        )
        assert response.rectified_query is None


class TestDocumentStatusResponse:
    """Test DocumentStatusResponse model"""
    
    def test_document_status_response_valid(self):
        """Test valid document status response"""
        response = DocumentStatusResponse(
            document_id="doc-123",
            status="completed",
            filename="test.pdf",
            chunks=5,
            entities=10
        )
        assert response.document_id == "doc-123"
        assert response.status == "completed"
        assert response.chunks == 5
        assert response.entities == 10
    
    def test_document_status_response_minimal(self):
        """Test document status response with minimal fields"""
        response = DocumentStatusResponse(
            document_id="doc-123",
            status="processing"
        )
        assert response.document_id == "doc-123"
        assert response.status == "processing"
        assert response.filename is None
        assert response.chunks is None
        assert response.entities is None
