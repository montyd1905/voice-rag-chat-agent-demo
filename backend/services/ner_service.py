import spacy
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. Please install it.")
    nlp = None


OUTER_CONFIDENCE_BOUND = 1.0
INNER_CONFIDENCE_BOUND = 0.7

class NERService:
    """Service for Named Entity Recognition"""
    
    @staticmethod
    def extract_entities(text: str) -> List[Dict]:
        """Extract named entities from text"""
        if not nlp:
            return []
        
        try:
            doc = nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "confidence": OUTER_CONFIDENCE_BOUND
                })
            
            return entities
        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            return []
    
    @staticmethod
    def extract_relationships(text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities"""
        
        relationships = []
        
        if not nlp or not entities:
            return relationships
        
        try:
            doc = nlp(text)
            
            for sent in doc.sents:
                sent_entities = [e for e in entities if e["start_char"] >= sent.start_char and e["end_char"] <= sent.end_char]
                
                if len(sent_entities) >= 2:
                    # create relationships between entities in the same sentence
                    for i in range(len(sent_entities) - 1):
                        relationships.append({
                            "subject": sent_entities[i]["text"],
                            "predicate": "co_occurs_with",
                            "object": sent_entities[i + 1]["text"],
                            "confidence": INNER_CONFIDENCE_BOUND
                        })
            
            return relationships
        except Exception as e:
            logger.error(f"Relationship extraction failed: {e}")
            return []

