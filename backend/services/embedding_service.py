from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging

logger = logging.getLogger(__name__)

# global model instance
_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Get or load the embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers"""
    
    @staticmethod
    def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
        """Generate embedding for a single text"""
        model = get_embedding_model(model_name)
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    @staticmethod
    def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        model = get_embedding_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

