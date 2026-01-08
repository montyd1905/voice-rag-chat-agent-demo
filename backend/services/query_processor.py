from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from services.ner_service import NERService
from services.embedding_service import EmbeddingService
from services.stt_service import STTService
from services.tts_service import TTSService
from database import get_connection
from redis_client import get_session_history, update_session_history, get_recent_qna_pairs
from config import settings
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-3.5-turbo"
MODEL_TEMPERATURE = 0.7
CONVERSATION_TURN_LIMIT = 3
COMMON_STOP_WORDS = ['the', 'and', 'or', 'what', 'who', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were']


class QueryProcessor:
    """Process user queries through the search pipeline following the design diagram"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    def process_text_query(self, query: str, session_id: str) -> Dict:
        """Process a text query"""
        return self._process_query(query, session_id, is_audio=False)
    
    def process_voice_query(self, audio_data: bytes, session_id: str) -> Dict:
        """Process a voice query - Step 1: STT"""
        logger.info(f"Transcribing audio for session {session_id}")
        query = STTService.transcribe(audio_data)
        
        if not query:
            return {
                "error": "Failed to transcribe audio",
                "response": None,
                "audio_response": None
            }
        
        # Process query
        result = self._process_query(query, session_id, is_audio=True)
        
        # Step: TTS if audio input
        if result.get("response") and not result.get("error") and result.get("is_audio"):
            logger.info(f"Generating audio response for session {session_id}")
            audio_response = TTSService.synthesize(result["response"])
            result["audio_response"] = audio_response
        
        return result
    
    def _process_query(self, query: str, session_id: str, is_audio: bool = False) -> Dict:
        """Main query processing pipeline following the design diagram"""
        try:
            # Step 1: Get conversation history
            history = get_session_history(session_id)
            conversation_history = history.get("conversation_history", []) if history else []
            
            # Step 2: Question Rectification (with conversation history)
            logger.info(f"Rectifying question for session {session_id}")
            contextually_updated_question = self._rectify_question(query, conversation_history)
            
            # Step 3: NER Filter
            logger.info(f"Extracting entities from rectified question: {contextually_updated_question}")
            entities = NERService.extract_entities(contextually_updated_question)
            logger.info(f"Extracted {len(entities)} entities: {[e.get('text', '') for e in entities]}")
            
            # If no entities extracted, try extracting from original query as fallback
            if not entities:
                logger.info("No entities from rectified query, trying original query")
                entities = NERService.extract_entities(query)
                logger.info(f"Extracted {len(entities)} entities from original: {[e.get('text', '') for e in entities]}")
            
            # Step 4: transform to Structured Query
            structured_query = self._create_structured_query(contextually_updated_question, entities)
            
            # Step 5: High-similarity search in recent QnA pairs
            logger.info(f"Searching recent QnA pairs for session {session_id}")
            qna_result = self._search_recent_qna(structured_query, session_id)
            
            # Decision Point 1: match found in recent QnA?
            if qna_result:
                logger.info(f"Match found in recent QnA pairs")
                resolved_result = qna_result["answer"]
                source = "cache"
            else:
                # Step 6: create embeddings with sentence-transformers
                logger.info(f"No match in QnA cache, searching vector database")
                query_embedding = EmbeddingService.generate_embedding(structured_query["rectified_question"])
                
                # Step 7: vector search in pgvector
                search_results = self._search_vector_db(query_embedding, structured_query)
                
                # Decision Point 2: result found in Vector DB?
                if search_results:
                    logger.info(f"Found {len(search_results)} results in vector database")
                    # extract search result
                    resolved_result = self._extract_search_result(search_results)
                    source = "vector_db"
                else:
                    logger.info(f"No results found in vector database")
                    resolved_result = "I couldn't find relevant information about that topic in the available documents."
                    source = "canned"
            
            # Step 8: final response generation
            final_response = self._generate_final_response(resolved_result, structured_query)
            
            # Step 9: update conversation history and recent QnA pairs
            update_session_history(session_id, query, final_response)
            
            # if result came from vector DB, ensure question embedding is stored in QnA cache
            if source == "vector_db":
                self._store_qna_embedding(session_id, structured_query["rectified_question"])
            
            return {
                "query": query,
                "rectified_query": contextually_updated_question,
                "response": final_response,
                "source": source,
                "entities": entities,
                "is_audio": is_audio,
                "audio_response": None 
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "error": str(e),
                "response": None,
                "audio_response": None
            }
    
    def _rectify_question(self, query: str, conversation_history: List[Dict]) -> str:
        """Question Rectification using LLM with conversation history"""
        try:
            # build conversation context
            context = ""
            if conversation_history:
                context = "\n".join([
                    f"User: {turn['user_query']}\nAssistant: {turn['system_response']}"
                    for turn in conversation_history[-CONVERSATION_TURN_LIMIT:]
                ])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a query rectification assistant. Generate a clear, simple question. Maximum 8 words. Only resolve pronouns if the query contains 'it', 'that', 'this', 'they', 'he', 'she'. Do not add explanations or extra context."),
                ("human", """Conversation History:
{history}

Current User Query: {query}

Generate a simple, direct question:""")
            ])
            
            if context:
                messages = prompt.format_messages(history=context, query=query)
            else:
                messages = prompt.format_messages(history="No previous conversation", query=query)
            
            response = self.llm.invoke(messages)
            rectified = response.content.strip().strip('"\'')
            
            # ensure it's a question
            if rectified and not rectified.endswith('?'):
                rectified = rectified.rstrip('.!') + '?'
            
            # limit length
            if len(rectified.split()) > 10:
                words = query.split()[:5]
                rectified = ' '.join(words) + '?'
            
            return rectified
            
        except Exception as e:
            logger.warning(f"Question rectification failed: {e}, using original query")
            return query if query.endswith('?') else query + '?'
    
    def _create_structured_query(self, rectified_question: str, entities: List[Dict]) -> Dict:
        """Transform rectified question and entities into structured query"""
        return {
            "rectified_question": rectified_question,
            "entities": entities,
            "query_embedding_ready": False
        }
    
    def _search_recent_qna(self, structured_query: Dict, session_id: str) -> Optional[Dict]:
        """High-similarity search in recent QnA pairs using embeddings"""
        qna_pairs = get_recent_qna_pairs(session_id)
        
        if not qna_pairs:
            return None
        
        # create embedding for rectified question
        query_embedding = EmbeddingService.generate_embedding(structured_query["rectified_question"])
        
        best_match = None
        best_similarity = 0.0
        
        for qna in qna_pairs:
            # get or create embedding for cached question
            if "question_embedding" in qna:
                qna_embedding = qna["question_embedding"]
            else:
                qna_embedding = EmbeddingService.generate_embedding(qna["question"])
                qna["question_embedding"] = qna_embedding
            
            # calculate cosine similarity
            similarity = np.dot(query_embedding, qna_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(qna_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = qna
        
        # decision: match found?
        if best_similarity >= settings.QNA_CACHE_THRESHOLD:
            return best_match
        
        return None
    
    def _search_vector_db(self, query_embedding: List[float], structured_query: Dict, top_k: int = 10) -> List[Dict]:
        """Vector search in pgvector database - semantic search with optional entity boost"""
        # convert embedding to string format for pgvector
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # extract entity texts for potential filtering (case-insensitive)
        entities = structured_query.get("entities", [])
        entity_texts = [e['text'].lower() for e in entities if e.get('text')] if entities else []
        
        # also extract key terms from the rectified question for text matching
        rectified_question = structured_query.get("rectified_question", "")
        # extract significant words from question (2+ characters, not common stop words)
        question_words = [w.lower() for w in rectified_question.split() 
                         if len(w) > 2 and w.lower() not in COMMON_STOP_WORDS]
        
        # combine entity texts and question words for text matching
        search_terms = list(set(entity_texts + question_words))
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                # primary search: semantic similarity with optional text matching boost
                # use a lower similarity threshold to catch more results, then filter/rank
                base_threshold = max(0.5, settings.SIMILARITY_THRESHOLD - 0.1)
                
                if search_terms:
                    # build text matching conditions (case-insensitive with partial matching)
                    text_conditions = []
                    text_params = []
                    for term in search_terms[:5]:  # limit to top 5 terms
                        # check in entities array and chunk_text (partial matches)
                        text_conditions.append(
                            "(EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'entities') AS elem WHERE LOWER(elem) LIKE %s) OR "
                            "LOWER(metadata->>'chunk_text') LIKE %s)"
                        )
                        # use LIKE with wildcards for partial matching in both places
                        text_params.append(f"%{term}%")  # Partial match in entities
                        text_params.append(f"%{term}%")  # Partial match in chunk_text
                    
                    # search with text matching as boost
                    sql_query = f"""
                        SELECT 
                            document_id,
                            chunk_id,
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity,
                            CASE WHEN ({' OR '.join(text_conditions)}) THEN 1 ELSE 0 END as text_match
                        FROM document_embeddings
                        WHERE 1 - (embedding <=> %s::vector) > %s
                        ORDER BY text_match DESC, embedding <=> %s::vector
                        LIMIT %s
                    """
                    
                    params = [embedding_str]
                    params.extend(text_params)
                    params.extend([embedding_str, base_threshold, embedding_str, top_k])
                else:
                    # no search terms, pure semantic search
                    sql_query = """
                        SELECT 
                            document_id,
                            chunk_id,
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity,
                            0 as text_match
                        FROM document_embeddings
                        WHERE 1 - (embedding <=> %s::vector) > %s
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    params = [embedding_str, embedding_str, base_threshold, embedding_str, top_k]
                
                cur.execute(sql_query, params)
                results = cur.fetchall()
                
                search_results = []
                text_matched_results = []
                
                for row in results:
                    similarity = float(row[3])
                    text_match = row[4] if len(row) > 4 else 0
                    
                    # if there's a text match (entity or keyword match), use lower threshold
                    if text_match == 1:
                        # accept text matches with lower similarity
                        if similarity >= 0.5:
                            text_matched_results.append({
                                "document_id": row[0],
                                "chunk_id": row[1],
                                "metadata": row[2],
                                "similarity": similarity
                            })
                    else:
                        # for non-text matches, use normal threshold
                        if similarity >= settings.SIMILARITY_THRESHOLD:
                            search_results.append({
                                "document_id": row[0],
                                "chunk_id": row[1],
                                "metadata": row[2],
                                "similarity": similarity
                            })
                
                # prioritize text-matched results
                if text_matched_results:
                    logger.info(f"Found {len(text_matched_results)} results with text/entity matches")
                    return text_matched_results[:top_k]
                
                # if we have regular results, return them
                if search_results:
                    logger.info(f"Found {len(search_results)} results from semantic search")
                    return search_results[:top_k]
                
                # if still no results, try with even lower threshold and text matching
                if search_terms:
                    logger.info(f"No results with threshold {settings.SIMILARITY_THRESHOLD}, trying with text matching and lower threshold")
                    # build text filter for WHERE clause (not just boost)
                    text_where_conditions = []
                    text_where_params = []
                    for term in search_terms[:3]:  # top 3 terms
                        text_where_conditions.append(
                            "(EXISTS (SELECT 1 FROM jsonb_array_elements_text(metadata->'entities') AS elem WHERE LOWER(elem) LIKE %s) OR "
                            "LOWER(metadata->>'chunk_text') LIKE %s)"
                        )
                        text_where_params.append(f"%{term}%")
                        text_where_params.append(f"%{term}%")
                    
                    fallback_query = f"""
                        SELECT 
                            document_id,
                            chunk_id,
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM document_embeddings
                        WHERE (1 - (embedding <=> %s::vector) > 0.4 OR ({' OR '.join(text_where_conditions)}))
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """
                    
                    fallback_params = [embedding_str, embedding_str]
                    fallback_params.extend(text_where_params)
                    fallback_params.extend([embedding_str, top_k])
                    
                    cur.execute(fallback_query, fallback_params)
                    results = cur.fetchall()
                    
                    for row in results:
                        search_results.append({
                            "document_id": row[0],
                            "chunk_id": row[1],
                            "metadata": row[2],
                            "similarity": float(row[3])
                        })
                else:
                    # pure semantic fallback
                    logger.info(f"No results with threshold {settings.SIMILARITY_THRESHOLD}, trying lower threshold 0.4")
                    cur.execute("""
                        SELECT 
                            document_id,
                            chunk_id,
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM document_embeddings
                        WHERE 1 - (embedding <=> %s::vector) > 0.4
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                    """, (embedding_str, embedding_str, embedding_str, top_k))
                    
                    results = cur.fetchall()
                    for row in results:
                        search_results.append({
                            "document_id": row[0],
                            "chunk_id": row[1],
                            "metadata": row[2],
                            "similarity": float(row[3])
                        })
                
                logger.info(f"Found {len(search_results)} results from vector database (fallback)")
                return search_results[:top_k]
    
    def _extract_search_result(self, search_results: List[Dict]) -> str:
        """Extract and format search result from vector database results"""
        if not search_results:
            return ""
        
        # combine top results
        result_texts = []
        for result in search_results[:3]:  # top 3 results
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            chunk_text = metadata.get("chunk_text", "")
            if chunk_text:
                result_texts.append(chunk_text)
        
        return "\n\n".join(result_texts)
    
    def _store_qna_embedding(self, session_id: str, question: str):
        """Store question embedding in recent QnA pairs for faster future searches"""
        try:
            history = get_session_history(session_id)
            if not history:
                return
            
            recent_qna = history.get("recent_qna_pairs", [])
            if recent_qna:
                # update the most recent entry with embedding if not already present
                latest_qna = recent_qna[-1]
                if "question_embedding" not in latest_qna:
                    question_embedding = EmbeddingService.generate_embedding(question)
                    latest_qna["question_embedding"] = question_embedding
                    
                    # save back to Redis
                    from redis_client import redis_client
                    if redis_client:
                        redis_client.setex(
                            f"session:{session_id}",
                            settings.SESSION_TTL,
                            json.dumps(history)
                        )
        except Exception as e:
            logger.warning(f"Failed to store QnA embedding: {e}")
    
    def _generate_final_response(self, resolved_result: str, structured_query: Dict) -> str:
        """Generate final response using LLM"""
        try:
            if not resolved_result or resolved_result.startswith("I couldn't find"):
                return resolved_result
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Answer the question using ONLY the information provided in the context. Do not add information not in the context."),
                ("human", """Context:
{context}

Question: {query}

Answer:""")
            ])
            
            messages = prompt.format_messages(
                context=resolved_result,
                query=structured_query["rectified_question"]
            )
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            # fallback to raw result
            return resolved_result
