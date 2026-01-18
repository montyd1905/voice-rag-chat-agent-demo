import pytest
from unittest.mock import patch, MagicMock
from services.document_processor import DocumentProcessor
import uuid


class TestDocumentProcessor:
    """Test DocumentProcessor service"""
    
    @pytest.fixture
    def document_processor(self):
        """Create DocumentProcessor instance"""
        return DocumentProcessor()
    
    @patch('services.document_processor.OCRService')
    @patch('services.document_processor.TFIDFService')
    @patch('services.document_processor.NERService')
    @patch('services.document_processor.EmbeddingService')
    @patch('services.document_processor.get_connection')
    def test_process_document_success(
        self, mock_get_conn, mock_embedding, mock_ner, mock_tfidf, mock_ocr, document_processor
    ):
        """Test successful document processing"""
        # Setup mocks
        mock_ocr.extract_text.return_value = "This is a test document about Rwanda. Rwanda is a country."
        mock_tfidf.denoise.return_value = "This is a test document about Rwanda. Rwanda is a country."
        mock_ner.extract_entities.return_value = [{"text": "Rwanda", "label": "GPE"}]
        mock_ner.extract_relationships.return_value = []
        mock_embedding.generate_embeddings.return_value = [[0.1] * 384, [0.2] * 384]
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        # Process document
        result = document_processor.process_document(
            b"fake pdf content",
            "test.pdf",
            "pdf"
        )
        
        assert "document_id" in result
        assert result["status"] == "completed"
        assert result["chunks"] > 0
        assert result["entities"] > 0
        mock_ocr.extract_text.assert_called_once()
        mock_tfidf.denoise.assert_called_once()
    
    @patch('services.document_processor.OCRService')
    @patch('services.document_processor.get_connection')
    def test_process_document_no_text(self, mock_get_conn, mock_ocr, document_processor):
        """Test document processing with no extracted text"""
        mock_ocr.extract_text.return_value = ""
        
        # Mock database connection for exception handler
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        with pytest.raises(ValueError, match="No text extracted"):
            document_processor.process_document(
                b"fake pdf content",
                "test.pdf",
                "pdf"
            )
    
    @patch('services.document_processor.OCRService')
    @patch('services.document_processor.get_connection')
    def test_process_document_short_text(self, mock_get_conn, mock_ocr, document_processor):
        """Test document processing with too short text"""
        mock_ocr.extract_text.return_value = "short"
        
        # Mock database connection for exception handler
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        with pytest.raises(ValueError, match="No text extracted"):
            document_processor.process_document(
                b"fake pdf content",
                "test.pdf",
                "pdf"
            )
    
    @patch('services.document_processor.OCRService')
    @patch('services.document_processor.TFIDFService')
    @patch('services.document_processor.NERService')
    @patch('services.document_processor.EmbeddingService')
    @patch('services.document_processor.get_connection')
    def test_process_document_processing_error(
        self, mock_get_conn, mock_embedding, mock_ner, mock_tfidf, mock_ocr, document_processor
    ):
        """Test document processing with error"""
        mock_ocr.extract_text.side_effect = Exception("OCR failed")
        
        with pytest.raises(Exception):
            document_processor.process_document(
                b"fake pdf content",
                "test.pdf",
                "pdf"
            )
    
    @patch('services.document_processor.get_connection')
    def test_store_embeddings(self, mock_get_conn, document_processor):
        """Test storing embeddings in database"""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        chunks = [
            {"chunk_id": "chunk1", "text": "Test chunk 1", "entities_mentioned": ["Rwanda"]}
        ]
        embeddings = [[0.1] * 384]
        metadata = {"document_id": "doc1", "source": "test.pdf"}
        
        document_processor._store_embeddings("doc1", chunks, embeddings, metadata)
        
        mock_cur.execute.assert_called()
        mock_conn.commit.assert_called_once()
    
    @patch('services.document_processor.get_connection')
    def test_update_document_status(self, mock_get_conn, document_processor):
        """Test updating document status"""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        document_processor._update_document_status(
            "doc1", "test.pdf", "pdf", 1024, "completed"
        )
        
        mock_cur.execute.assert_called()
        mock_conn.commit.assert_called_once()
