import uuid
from typing import Dict, List
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.ocr_service import OCRService
from services.tfidf_service import TFIDFService
from services.ner_service import NERService
from services.embedding_service import EmbeddingService
from database import get_connection
from config import settings
import json
import logging

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process documents through the indexing pipeline"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def process_document(self, file_data: bytes, filename: str, file_type: str) -> Dict:
        """Process a document through the full pipeline"""
        document_id = str(uuid.uuid4())
        
        try:
            # Step 1: OCR - extract text
            logger.info(f"Processing document {document_id}: OCR")
            raw_text = OCRService.extract_text(file_data, file_type)
            
            if not raw_text or len(raw_text.strip()) < 10:
                raise ValueError("No text extracted from document")
            
            # Step 2: TF-IDF Filter - denoise
            logger.info(f"Processing document {document_id}: TF-IDF filtering")
            cleaned_text = TFIDFService.denoise(raw_text)
            
            # Step 3: split into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Step 4: process each chunk
            all_entities = []
            all_relationships = []
            chunk_data = []
            
            for i, chunk in enumerate(chunks):
                # NER extraction
                entities = NERService.extract_entities(chunk)
                relationships = NERService.extract_relationships(chunk, entities)
                
                all_entities.extend(entities)
                all_relationships.extend(relationships)
                
                chunk_data.append({
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "text": chunk,
                    "entities_mentioned": [e["text"] for e in entities]
                })
            
            # Step 5: create structured JSON
            structured_knowledge = {
                "document_id": document_id,
                "source": filename,
                "extraction_timestamp": datetime.now().isoformat(),
                "entities": all_entities,
                "relationships": all_relationships,
                "original_text_chunks": chunk_data
            }
            
            # Step 6: generate embeddings and store
            logger.info(f"Processing document {document_id}: Generating embeddings")
            chunk_texts = [chunk["text"] for chunk in chunk_data]
            embeddings = EmbeddingService.generate_embeddings(chunk_texts)
            
            # Step 7: store in database
            logger.info(f"Processing document {document_id}: Storing in database")
            self._store_embeddings(document_id, chunk_data, embeddings, structured_knowledge)
            
            # update document status
            self._update_document_status(document_id, filename, file_type, len(file_data), "completed")
            
            return {
                "document_id": document_id,
                "status": "completed",
                "chunks": len(chunks),
                "entities": len(all_entities)
            }
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            self._update_document_status(document_id, filename, file_type, len(file_data), "failed")
            raise
    
    def _store_embeddings(self, document_id: str, chunks: List[Dict], embeddings: List[List[float]], metadata: Dict):
        """Store embeddings in the database"""
        with get_connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    # convert embedding list to string format for pgvector
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    cur.execute("""
                        INSERT INTO document_embeddings 
                        (document_id, chunk_id, embedding, metadata)
                        VALUES (%s, %s, %s::vector, %s::jsonb)
                    """, (
                        document_id,
                        chunk["chunk_id"],
                        embedding_str,
                        json.dumps({
                            "chunk_text": chunk["text"],
                            "entities": chunk["entities_mentioned"],
                            "document_metadata": metadata
                        })
                    ))
                conn.commit()
    
    def _update_document_status(self, document_id: str, filename: str, file_type: str, file_size: int, status: str):
        """Update document status in database"""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents (document_id, filename, file_type, file_size, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (document_id) 
                    DO UPDATE SET status = %s, updated_at = NOW()
                """, (document_id, filename, file_type, file_size, status, status))
                conn.commit()

