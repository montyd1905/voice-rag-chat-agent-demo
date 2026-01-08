from psycopg2.pool import SimpleConnectionPool
from contextlib import contextmanager
from config import settings
import logging

logger = logging.getLogger(__name__)

# connection pool
pool = None


def init_db():
    """Initialize database connection pool and create tables"""
    global pool
    
    pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        database=settings.POSTGRES_DB
    )
    
    # create tables
    with get_connection() as conn:
        with conn.cursor() as cur:
            # enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # create document_embeddings table (all-MiniLM-L6-v2 produces 384-dimensional embeddings)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    chunk_id VARCHAR(255),
                    embedding vector(384),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # create index if not exists
            cur.execute("""
                CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
                ON document_embeddings 
                USING hnsw (embedding vector_cosine_ops);
            """)
            
            # create documents table for metadata
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255),
                    file_type VARCHAR(50),
                    file_size BIGINT,
                    status VARCHAR(50) DEFAULT 'processing',
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")


@contextmanager
def get_connection():
    """Get a connection from the pool"""
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


def close_pool():
    """Close all connections in the pool"""
    if pool:
        pool.closeall()

