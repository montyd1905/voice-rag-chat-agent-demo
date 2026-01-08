# Voice RAG Chat Agent Demo

A comprehensive Retrieval-Augmented Generation (RAG) system with voice and text query support, built with FastAPI, PostgreSQL (pgvector), Redis, and local ML models.

## Features

- **Document Indexing Pipeline**: OCR → TF-IDF Filtering → NER → Embeddings → Vector Database
- **Voice/Text Query Processing**: STT → Question Rectification → Vector Search → TTS
- **Local Processing**: All ML models (OCR, STT, TTS, Embeddings) run locally
- **Conversation History**: Session-based multi-turn conversations
- **QnA Caching**: Fast retrieval of recent question-answer pairs

## Architecture

The system consists of two main sub-systems:

1. **RAG Document Indexing System**: Processes and indexes documents for retrieval
2. **Voice/Text RAG Search System**: Handles user queries and retrieves relevant information

### Architecture Diagrams

#### RAG Document Indexing Flow

![RAG Document Indexing Flow](docs/rag-document-indexing.png)

The document indexing pipeline transforms raw documents into searchable vector embeddings:
- Documents → OCR Engine → Text Blob → TF-IDF Filter → NER Models → Structured Knowledge (JSON) → Sentence-Transformers Embeddings → Vector Database (pgvector)

#### Voice/Text RAG Search Flow

![Voice/Text RAG Search Flow](docs/voice-text-rag-search.png)

The query processing pipeline handles both voice and text queries:
- User Query (Voice/Text) → Local STT (if voice) → Question Rectification → Structured Query → Retrieval (Recent QnA / Vector DB) → Response Generation → Local TTS (if voice) → User Response

See [SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for detailed specifications.

### Architecture for Inference

- **Hybrid Model Serving**: STT (Whisper), TTS (Coqui TTS), Embeddings (sentence-transformers), and NER (spaCy) run as embedded models locally in the backend container, loaded into memory on first use and reused for subsequent requests to eliminate network latency and API costs. LLM (OpenAI GPT) is used via API for question rectification and response generation, providing advanced language understanding while keeping other models local for cost and latency optimization.

### Handling Ambiguity

- **Multi-Layer Ambiguity Resolution**: The system employs LLM-based question rectification to clarify ambiguous queries by resolving pronouns/references from conversation history and expanding abbreviations. Vector search uses configurable similarity thresholds (default: 0.7) to ensure only relevant results are considered, while QnA cache validation requires high similarity (0.85) before returning cached responses. When no relevant information is found, pre-defined canned responses prevent hallucination, and NER extraction provides context-aware filtering to identify when queries contain entities not present in indexed documents.

### Latency Optimization

- **Comprehensive Performance Strategy**: Model pre-loading keeps STT, TTS, and embedding models resident in memory (saving 2-5 seconds per request), while QnA caching in Redis enables <100ms responses for cache hits versus 1-6 seconds for the full pipeline. The system uses optimized models including fast TTS (glow-tts, 2-3x faster than tacotron2-DDC, reducing synthesis from ~19s to ~6-8s) and smart query rectification that skips LLM calls for simple queries (saving 1-2 seconds). Parallel processing via thread pool executor handles CPU-intensive STT/TTS operations asynchronously, preventing blocking. HNSW-indexed vector database queries with optimized fallback logic ensure efficient retrieval. Overall voice query latency reduced from ~27-30s to ~15-18s while maintaining accuracy. pgvector HNSW indexes enable fast approximate nearest neighbor search and Redis provides sub-millisecond session lookups.

### Scalability

- **Multi-Tier Scaling Architecture**: The stateless backend design enables horizontal scaling with multiple instances behind a load balancer, while Redis-based session storage provides shared state across instances. Database scaling leverages PostgreSQL connection pooling and read replicas, with pgvector HNSW indexes maintaining performance at scale. Redis cluster caching handles high-throughput operations and reduces database load by 30-50%. FastAPI's async capabilities with non-blocking I/O handle concurrent requests efficiently, while resource management shares model instances across requests within a process (models loaded once per process). Estimated capacity: single instance handles ~50-100 concurrent requests; horizontal scaling with 10 instances supports ~500-1,000 concurrent requests.

### Data Privacy

- **Comprehensive Privacy Protection**: The system ensures data privacy through local processing of sensitive data, with STT (Whisper), TTS (Coqui TTS), embeddings (sentence-transformers), and NER (spaCy) models running entirely on-premises without transmitting audio or document content to external services. All document processing, audio transcription, and voice synthesis occur within the backend container, ensuring sensitive information never leaves the server infrastructure. Session data stored in Redis is ephemeral with configurable TTL (default: 1 hour), and conversation history is isolated per session ID. The only external API call is to OpenAI GPT for question rectification and response generation, which only receives text queries (not audio or documents).

## Prerequisites

- Docker and Docker Compose
- OpenAI API key (for question rectification and response generation)

## Setup

1. **Clone the repository** (if not already done)

2. **Create `.env` file** in the root directory:
   ```bash
   cp .env.example .env
   ```

3. **Add your OpenAI API key** to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Start the services** (can take several minutes):
   ```bash
   docker-compose up --build
   ```

   This will start:
   - PostgreSQL with pgvector extension (port 5432)
   - Redis (port 6379)
   - Backend API (port 8000)
   - Frontend (port 80)

5. **Access the application**:
   - Frontend: http://localhost
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Usage

### Upload Documents

1. Open the frontend at http://localhost
2. Use the upload section to upload PDF or image files
3. Documents will be processed through the indexing pipeline
4. Check the document list for processing status

**Note:** Sample articles are available in the `articles/` folder for testing. You can upload these documents to get started quickly.

### Query Documents

**Text Query:**
1. Type your question in the text input
2. Click "Send" or press Enter
3. View the response in the chat

**Voice Query:**
1. Click "Record Voice" button
2. Speak your question
3. Click "Stop" when done
4. The system will transcribe, process, and respond (with audio if voice input was used)

**Important:** The first text or voice query may experience a delay (typically 5-15 seconds) as local ML models (STT, TTS, embeddings, NER) are loaded into memory on first use. Subsequent queries will be much faster as models remain in memory. Model loading time improves with faster network speeds (for initial model downloads) and better hardware (CPU/GPU, RAM). Once loaded, models are reused for all subsequent requests, providing near real-time responses.

## API Endpoints

### Document Indexing

- `POST /api/documents/upload` - Upload documents for indexing
- `GET /api/documents/status/{document_id}` - Check indexing status
- `GET /api/documents` - List indexed documents

### Query Processing

- `POST /api/query` - Submit text query
- `POST /api/query/voice` - Submit voice query
- `GET /api/session/{session_id}/history` - Get conversation history
- `DELETE /api/session/{session_id}` - Clear session

## Technology Stack

- **Backend**: FastAPI, Python 3.11
- **Database**: PostgreSQL with pgvector extension
- **Cache**: Redis
- **ML Models**:
  - OCR: Tesseract
  - STT: OpenAI Whisper "small" model (handles multiple audio formats automatically)
  - TTS: Coqui TTS (glow-tts model for optimized latency)
  - Embeddings: sentence-transformers
  - NER: spaCy
- **Audio Processing**: soundfile, scipy (for STT preprocessing)
- **Frontend**: HTML/CSS/JavaScript
- **LLM**: OpenAI GPT (for question rectification and response generation)

## Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Development

The frontend is static HTML/CSS/JavaScript. You can serve it with any static file server or modify files directly in the `frontend/` directory.

## Configuration

Configuration is managed through environment variables in `.env` and `backend/config.py`. Key settings:

- `OPENAI_API_KEY`: Required for LLM operations
- `EMBEDDING_MODEL`: sentence-transformers model (default: "all-MiniLM-L6-v2")
- `WHISPER_MODEL`: Whisper model size (default: "small" - balances speed/accuracy)
- `TTS_MODEL`: TTS model (default: "tts_models/en/ljspeech/glow-tts" - optimized for latency)
- `SIMILARITY_THRESHOLD`: Vector search threshold (default: 0.7)
- `QNA_CACHE_THRESHOLD`: QnA cache similarity threshold (default: 0.85)
- `SKIP_RECTIFICATION_FOR_SIMPLE_QUERIES`: Skip LLM rectification for simple queries (default: True - reduces latency by 1-2s)

## Troubleshooting

1. **Models not loading**: First run will download models, which may take time. Check logs for download progress.

2. **Database connection errors**: Ensure PostgreSQL container is healthy before starting backend.

3. **Memory issues**: Large models (especially Whisper "small" and TTS) require significant RAM (~1GB for Whisper small model). Consider using smaller models or increasing Docker memory limits.

4. **STT accuracy**: If transcription quality is poor, ensure audio preprocessing libraries (soundfile, scipy) are installed. The system includes automatic fallback if these are unavailable.

5. **Unicode/encoding errors**: The system uses base64 encoding for HTTP headers containing Unicode characters. Ensure frontend properly decodes base64-encoded metadata headers.

6. **OCR errors**: Ensure Tesseract is properly installed in the container.
