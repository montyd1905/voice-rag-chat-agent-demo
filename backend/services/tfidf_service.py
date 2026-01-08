import re
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import logging

logger = logging.getLogger(__name__)


class TFIDFService:
    """Service for TF-IDF filtering and denoising"""
    
    # English stop words
    STOP_WORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
        'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
        'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
        'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
        'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
        'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
        'come', 'made', 'may', 'part'
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
        # normalize to lowercase
        text = text.lower()
        return text.strip()
    
    @staticmethod
    def filter_by_tfidf(text: str, threshold: float = 0.01) -> str:
        """Filter text using TF-IDF to remove low-value terms"""
        try:
            # split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return text
            
            # calculate TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words=list(TFIDFService.STOP_WORDS),
                min_df=1,
                max_df=0.95
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # filter sentences with meaningful content
                filtered_sentences = []
                for i, sentence in enumerate(sentences):
                    # get TF-IDF scores for this sentence
                    scores = tfidf_matrix[i].toarray()[0]
                    max_score = scores.max() if len(scores) > 0 else 0
                    
                    # keep sentence if it has meaningful terms
                    if max_score > threshold:
                        filtered_sentences.append(sentence)
                
                return ' '.join(filtered_sentences)
            except ValueError:
                # if TF-IDF fails (e.g., all stop words), return cleaned text
                return TFIDFService.clean_text(text)
        except Exception as e:
            logger.warning(f"TF-IDF filtering failed: {e}, returning cleaned text")
            return TFIDFService.clean_text(text)
    
    @staticmethod
    def denoise(text: str) -> str:
        """Main denoising function"""
        # clean text
        cleaned = TFIDFService.clean_text(text)
        # apply TF-IDF filtering
        filtered = TFIDFService.filter_by_tfidf(cleaned)
        return filtered

