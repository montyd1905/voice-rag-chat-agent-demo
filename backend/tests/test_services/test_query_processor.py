import pytest
from unittest.mock import patch, MagicMock, Mock
from services.query_processor import QueryProcessor
import numpy as np


class TestQueryProcessor:
    """Test QueryProcessor service"""
    
    @pytest.fixture
    def query_processor(self):
        """Create QueryProcessor instance with mocked LLM"""
        with patch('services.query_processor.ChatOpenAI'):
            return QueryProcessor()
    
    def test_is_simple_query(self, query_processor):
        """Test simple query detection"""
        # Note: The logic checks if any COMMON_STOP_WORDS are IN the query string
        # "What" is in COMMON_STOP_WORDS, so queries containing "what" are not simple
        # Simple query - short, no stop words, ends with ?, no history
        # "Capital?" should work: short (1 word), no stop words, ends with ?, no history
        assert query_processor._is_simple_query("Capital?", []) == True
        
        # Query with stop words (pronouns/references)
        assert query_processor._is_simple_query("What is it?", []) == False
        
        # Query with "the" (in COMMON_STOP_WORDS)
        assert query_processor._is_simple_query("The capital?", []) == False
        
        # Long query
        long_query = "What is Rwanda and what are its main characteristics and features?"
        assert query_processor._is_simple_query(long_query, []) == False
        
        # Query with conversation history
        history = [{"user_query": "Hello", "system_response": "Hi"}]
        assert query_processor._is_simple_query("What is it?", history) == False
    
    @patch('services.query_processor.ChatOpenAI')
    def test_rectify_question(self, mock_llm_class):
        """Test question rectification"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "What is Rwanda?"
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        processor = QueryProcessor()
        result = processor._rectify_question("Rwanda?", [])
        
        assert result.endswith("?")
        assert len(result.split()) <= 10
    
    @patch('services.query_processor.ChatOpenAI')
    def test_rectify_question_with_history(self, mock_llm_class):
        """Test question rectification with conversation history"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "What is the capital of Rwanda?"
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        processor = QueryProcessor()
        history = [
            {"user_query": "What is Rwanda?", "system_response": "Rwanda is a country."}
        ]
        result = processor._rectify_question("What is its capital?", history)
        
        assert result.endswith("?")
        mock_llm.invoke.assert_called_once()
    
    def test_create_structured_query(self, query_processor):
        """Test structured query creation"""
        entities = [{"text": "Rwanda", "label": "GPE"}]
        result = query_processor._create_structured_query("What is Rwanda?", entities)
        
        assert result["rectified_question"] == "What is Rwanda?"
        assert result["entities"] == entities
        assert result["query_embedding_ready"] == False
    
    @patch('services.query_processor.EmbeddingService')
    @patch('services.query_processor.get_recent_qna_pairs')
    def test_search_recent_qna_found(self, mock_get_qna, mock_embedding_service, query_processor):
        """Test searching recent QnA pairs with match found"""
        mock_get_qna.return_value = [
            {
                "question": "What is Rwanda?",
                "answer": "Rwanda is a country.",
                "question_embedding": np.array([0.1] * 384)
            }
        ]
        
        mock_embedding = np.array([0.1] * 384)
        mock_embedding_service.generate_embedding.return_value = mock_embedding
        
        structured_query = {
            "rectified_question": "What is Rwanda?",
            "entities": []
        }
        
        result = query_processor._search_recent_qna(structured_query, "test-session")
        
        assert result is not None
        assert result["answer"] == "Rwanda is a country."
    
    @patch('services.query_processor.EmbeddingService')
    @patch('services.query_processor.get_recent_qna_pairs')
    def test_search_recent_qna_not_found(self, mock_get_qna, mock_embedding_service, query_processor):
        """Test searching recent QnA pairs with no match"""
        # Create embeddings that are very different (low similarity)
        # Use opposite values to ensure low cosine similarity
        mock_get_qna.return_value = [
            {
                "question": "What is Kenya?",
                "answer": "Kenya is a country.",
                "question_embedding": np.array([1.0] * 384)  # All 1.0s
            }
        ]
        
        # Query embedding with all -1.0s (opposite direction)
        mock_embedding = np.array([-1.0] * 384)
        mock_embedding_service.generate_embedding.return_value = mock_embedding
        
        structured_query = {
            "rectified_question": "What is Rwanda?",
            "entities": []
        }
        
        # Mock the threshold to be high so similarity check fails
        with patch('services.query_processor.settings') as mock_settings:
            mock_settings.QNA_CACHE_THRESHOLD = 0.9
            result = query_processor._search_recent_qna(structured_query, "test-session")
            assert result is None
    
    @patch('services.query_processor.EmbeddingService')
    @patch('services.query_processor.get_connection')
    def test_search_vector_db(self, mock_get_conn, mock_embedding_service, query_processor):
        """Test vector database search"""
        mock_embedding = [0.1] * 384
        mock_embedding_service.generate_embedding.return_value = mock_embedding
        
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            ("doc1", "chunk1", '{"chunk_text": "Rwanda is a country"}', 0.85, 0)
        ]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_conn.cursor.return_value.__exit__.return_value = None
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_get_conn.return_value.__exit__.return_value = None
        
        structured_query = {
            "rectified_question": "What is Rwanda?",
            "entities": []
        }
        
        results = query_processor._search_vector_db(mock_embedding, structured_query)
        
        assert len(results) > 0
        assert results[0]["similarity"] == 0.85
    
    def test_extract_search_result(self, query_processor):
        """Test extracting search result from vector DB results"""
        search_results = [
            {
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "metadata": {"chunk_text": "Rwanda is a country in East Africa."},
                "similarity": 0.85
            },
            {
                "document_id": "doc1",
                "chunk_id": "chunk2",
                "metadata": {"chunk_text": "The capital is Kigali."},
                "similarity": 0.80
            }
        ]
        
        result = query_processor._extract_search_result(search_results)
        
        assert "Rwanda is a country" in result
        assert "The capital is Kigali" in result
    
    @patch('services.query_processor.ChatOpenAI')
    def test_generate_final_response(self, mock_llm_class):
        """Test final response generation"""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Rwanda is a country in East Africa."
        mock_llm.invoke.return_value = mock_response
        mock_llm_class.return_value = mock_llm
        
        processor = QueryProcessor()
        structured_query = {"rectified_question": "What is Rwanda?"}
        resolved_result = "Rwanda is a country in East Africa."
        
        result = processor._generate_final_response(resolved_result, structured_query)
        
        assert "Rwanda" in result
        mock_llm.invoke.assert_called_once()
    
    @patch('services.query_processor.ChatOpenAI')
    def test_generate_final_response_no_result(self, mock_llm_class):
        """Test final response generation with no result"""
        processor = QueryProcessor()
        structured_query = {"rectified_question": "What is XYZ?"}
        resolved_result = "I couldn't find relevant information about that topic in the available documents."
        
        result = processor._generate_final_response(resolved_result, structured_query)
        
        assert result == resolved_result
        mock_llm_class.return_value.invoke.assert_not_called()
    
    @patch('services.query_processor.STTService')
    @patch('services.query_processor.TTSService')
    @patch('services.query_processor.QueryProcessor._process_query')
    def test_process_voice_query(self, mock_process, mock_tts, mock_stt, query_processor):
        """Test voice query processing"""
        mock_stt.transcribe.return_value = "What is Rwanda?"
        mock_process.return_value = {
            "query": "What is Rwanda?",
            "response": "Rwanda is a country.",
            "is_audio": True
        }
        mock_tts.synthesize.return_value = b"fake audio"
        
        result = query_processor.process_voice_query(b"fake audio data", "test-session")
        
        assert result["query"] == "What is Rwanda?"
        assert result["audio_response"] == b"fake audio"
        mock_stt.transcribe.assert_called_once()
        mock_tts.synthesize.assert_called_once()
    
    @patch('services.query_processor.STTService')
    def test_process_voice_query_transcription_failed(self, mock_stt, query_processor):
        """Test voice query with failed transcription"""
        mock_stt.transcribe.return_value = None
        
        result = query_processor.process_voice_query(b"fake audio data", "test-session")
        
        assert "error" in result
        assert result["error"] == "Failed to transcribe audio"
