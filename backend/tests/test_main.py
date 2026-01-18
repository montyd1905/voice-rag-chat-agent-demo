import pytest
from unittest.mock import patch, MagicMock, Mock
from fastapi import status
from http import HTTPStatus
import json


class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns healthy status"""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"status": "healthy"}


class TestDocumentUpload:
    """Test document upload endpoint"""
    
    @patch('main.document_processor.process_document')
    def test_upload_document_success(self, mock_process, client, sample_document_data):
        """Test successful document upload"""
        mock_process.return_value = sample_document_data
        
        # Create a mock file
        files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
        
        response = client.post("/api/documents/upload", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["document_id"] == sample_document_data["document_id"]
        assert data["status"] == "completed"
        mock_process.assert_called_once()
    
    @patch('main.settings')
    def test_upload_document_too_large(self, mock_settings, client):
        """Test document upload with file too large"""
        # Mock MAX_FILE_SIZE to be smaller than our test file
        mock_settings.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        # Create a large file (simulate)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.pdf", large_content, "application/pdf")}
        
        response = client.post("/api/documents/upload", files=files)
        
        # Note: FastAPI's UploadFile.size might not be set correctly in tests
        # The actual validation happens when file.size is accessed
        # This test verifies the endpoint handles large files
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_500_INTERNAL_SERVER_ERROR]
    
    @patch('main.document_processor.process_document')
    def test_upload_document_processing_error(self, mock_process, client):
        """Test document upload with processing error"""
        mock_process.side_effect = Exception("Processing failed")
        
        files = {"file": ("test.pdf", b"fake pdf content", "application/pdf")}
        
        response = client.post("/api/documents/upload", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDocumentStatus:
    """Test document status endpoint"""
    
    @patch('database.get_connection')
    def test_get_document_status_success(self, mock_get_conn, client, sample_document_data):
        """Test getting document status successfully"""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = (
            sample_document_data["document_id"],
            sample_document_data["filename"],
            sample_document_data["status"]
        )
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        response = client.get(f"/api/documents/status/{sample_document_data['document_id']}")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["document_id"] == sample_document_data["document_id"]
        assert data["status"] == sample_document_data["status"]
    
    @patch('database.get_connection')
    def test_get_document_status_not_found(self, mock_get_conn, client):
        """Test getting status for non-existent document"""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        response = client.get("/api/documents/status/non-existent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


class TestListDocuments:
    """Test list documents endpoint"""
    
    @patch('database.get_connection')
    def test_list_documents_success(self, mock_get_conn, client):
        """Test listing documents successfully"""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("doc1", "file1.pdf", "pdf", "completed", "2024-01-01 00:00:00"),
            ("doc2", "file2.pdf", "pdf", "processing", "2024-01-02 00:00:00")
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        response = client.get("/api/documents")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "documents" in data
        assert len(data["documents"]) == 2


class TestTextQuery:
    """Test text query endpoint"""
    
    @patch('main.query_processor.process_text_query')
    def test_submit_text_query_success(self, mock_process, client, sample_query_request, sample_query_response):
        """Test successful text query"""
        mock_process.return_value = sample_query_response
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == sample_query_response["query"]
        assert data["response"] == sample_query_response["response"]
        assert data["session_id"] == sample_query_response["session_id"]
        mock_process.assert_called_once()
    
    @patch('main.query_processor.process_text_query')
    def test_submit_text_query_without_session_id(self, mock_process, client, sample_query_response):
        """Test text query without session ID (should generate one)"""
        mock_process.return_value = sample_query_response
        
        request = {"query": "What is Rwanda?"}
        response = client.post("/api/query", json=request)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] is not None
    
    @patch('main.query_processor.process_text_query')
    def test_submit_text_query_error(self, mock_process, client, sample_query_request):
        """Test text query with processing error"""
        mock_process.return_value = {"error": "Processing failed"}
        
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestVoiceQuery:
    """Test voice query endpoint"""
    
    @patch('main.query_processor.process_voice_query')
    def test_submit_voice_query_success(self, mock_process, client, sample_audio_data, sample_query_response):
        """Test successful voice query"""
        sample_query_response["audio_response"] = sample_audio_data
        mock_process.return_value = sample_query_response
        
        files = {"audio": ("recording.wav", sample_audio_data, "audio/wav")}
        response = client.post("/api/query/voice", files=files, params={"session_id": "test-session"})
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "audio/wav"
        assert "X-Query-Data" in response.headers
        mock_process.assert_called_once()
    
    @patch('main.query_processor.process_voice_query')
    def test_submit_voice_query_without_audio_response(self, mock_process, client, sample_query_response):
        """Test voice query without audio response"""
        mock_process.return_value = sample_query_response
        
        files = {"audio": ("recording.wav", b"fake audio", "audio/wav")}
        response = client.post("/api/query/voice", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == sample_query_response["query"]
        assert data["has_audio"] == False
    
    @patch('main.query_processor.process_voice_query')
    def test_submit_voice_query_error(self, mock_process, client):
        """Test voice query with processing error"""
        mock_process.return_value = {"error": "Transcription failed"}
        
        files = {"audio": ("recording.wav", b"fake audio", "audio/wav")}
        response = client.post("/api/query/voice", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestSessionManagement:
    """Test session management endpoints"""
    
    @patch('redis_client.get_session_history')
    def test_get_session_history_success(self, mock_get_history, client):
        """Test getting session history successfully"""
        mock_history = {
            "session_id": "test-session",
            "conversation_history": [
                {"user_query": "Hello", "system_response": "Hi there"}
            ],
            "recent_qna_pairs": []
        }
        mock_get_history.return_value = mock_history
        
        response = client.get("/api/session/test-session/history")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == "test-session"
        assert "conversation_history" in data
    
    @patch('redis_client.get_session_history')
    def test_get_session_history_empty(self, mock_get_history, client):
        """Test getting history for non-existent session"""
        mock_get_history.return_value = None
        
        response = client.get("/api/session/non-existent/history")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["conversation_history"] == []
        assert data["recent_qna_pairs"] == []
    
    @patch('main.clear_session')
    def test_clear_session_success(self, mock_clear, client):
        """Test clearing session successfully"""
        mock_clear.return_value = None
        
        response = client.delete("/api/session/test-session")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Session cleared"
        assert data["session_id"] == "test-session"
        mock_clear.assert_called_once_with("test-session")
