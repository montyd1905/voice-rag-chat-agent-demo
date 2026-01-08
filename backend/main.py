from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import uuid
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus

from config import settings
from database import init_db, close_pool
from redis_client import init_redis, get_session_history, clear_session
from services.document_processor import DocumentProcessor
from services.query_processor import QueryProcessor
from models import QueryRequest, QueryResponse, DocumentStatusResponse

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice RAG Chat Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# thread pool for async document processing
executor = ThreadPoolExecutor(max_workers=4)

# initialize services
document_processor = DocumentProcessor()
query_processor = QueryProcessor()


@app.on_event("startup")
async def startup_event():
    """Initialize database and Redis on startup"""
    logger.info("Initializing services...")
    init_db()
    init_redis()
    logger.info("Services initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    close_pool()
    logger.info("Services shut down")


# =========================================== Document Indexing ===========================================

@app.post("/api/documents/upload", response_model=DocumentStatusResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # validate file
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="File too large")
        
        # read file data
        file_data = await file.read()
        file_type = file.filename.split('.')[-1] if '.' in file.filename else 'unknown'
        
        # process document in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            document_processor.process_document,
            file_data,
            file.filename,
            file_type
        )
        
        return DocumentStatusResponse(**result)
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/documents/status/{document_id}", response_model=DocumentStatusResponse)
async def get_document_status(document_id: str):
    """Get document processing status"""
    from database import get_connection
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_id, filename, status
                    FROM documents
                    WHERE document_id = %s
                """, (document_id,))
                
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Document not found")
                
                return DocumentStatusResponse(
                    document_id=row[0],
                    status=row[2],
                    filename=row[1]
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List all documents"""
    from database import get_connection
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document_id, filename, file_type, status, created_at
                    FROM documents
                    ORDER BY created_at DESC
                    LIMIT 100
                """)
                
                rows = cur.fetchall()
                documents = []
                for row in rows:
                    documents.append({
                        "document_id": row[0],
                        "filename": row[1],
                        "file_type": row[2],
                        "status": row[3],
                        "created_at": str(row[4])
                    })
                
                return {"documents": documents}
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


# =========================================== Query Processing ===========================================

@app.post("/api/query", response_model=QueryResponse)
async def submit_text_query(request: QueryRequest):
    """Submit a text query"""
    try:
        # generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # process query
        result = query_processor.process_text_query(request.query, session_id)
        
        if result.get("error"):
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result["error"])
        
        return QueryResponse(
            query=result["query"],
            rectified_query=result.get("rectified_query"),
            response=result["response"],
            source=result["source"],
            entities=result.get("entities", []),
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/query/voice")
async def submit_voice_query(
    audio: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Submit a voice query"""
    try:
        # generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # read audio data
        audio_data = await audio.read()
        
        # process query
        result = query_processor.process_voice_query(audio_data, session_id)
        
        if result.get("error"):
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result["error"])
        
        # return response with audio if available
        response_data = {
            "query": result["query"],
            "rectified_query": result.get("rectified_query"),
            "response": result["response"],
            "source": result["source"],
            "entities": result.get("entities", []),
            "session_id": session_id,
            "has_audio": result.get("audio_response") is not None
        }
        
        # if audio response exists, return it as streaming response
        if result.get("audio_response"):
            # Encode JSON data as base64 to avoid latin-1 codec issues with Unicode
            import base64
            import json as json_lib
            json_str = json_lib.dumps(response_data, ensure_ascii=False)
            encoded_data = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
            
            return StreamingResponse(
                iter([result["audio_response"]]),
                media_type="audio/wav",
                headers={
                    "X-Query-Data": encoded_data,
                    "X-Query-Data-Encoding": "base64"
                }
            )
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query processing failed: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


# =========================================== Session Management ===========================================

@app.get("/api/session/{session_id}/history")
async def get_session_history_endpoint(session_id: str):
    """Get conversation history for a session"""
    try:
        history = get_session_history(session_id)
        if not history:
            return {"session_id": session_id, "conversation_history": [], "recent_qna_pairs": []}
        return history
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.delete("/api/session/{session_id}")
async def clear_session_endpoint(session_id: str):
    """Clear a session"""
    try:
        clear_session(session_id)
        return {"message": "Session cleared", "session_id": session_id}
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

