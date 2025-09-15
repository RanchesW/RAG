import re
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class RussianNLP:
    def __init__(self):
        try:
            # Try to import optional Russian libraries
            try:
                import pymorphy3
                self.morph = pymorphy3.MorphAnalyzer()
            except ImportError:
                self.morph = None
                logger.warning("pymorphy3 not available")
            
            try:
                import razdel
                self.razdel = razdel
            except ImportError:
                self.razdel = None
                logger.warning("razdel not available")
            
            try:
                from nltk.corpus import stopwords
                self.russian_stopwords = set(stopwords.words('russian'))
            except:
                self.russian_stopwords = set(['и', 'в', 'на', 'с', 'по', 'для', 'от', 'к', 'из'])
                logger.warning("NLTK stopwords not available, using basic set")
            
            logger.info("✓ Russian NLP components loaded")
        except Exception as e:
            logger.error(f"Failed to load Russian NLP: {e}")
            self.morph = None
            self.razdel = None
            self.russian_stopwords = set()
    
    def normalize_text(self, text: str) -> str:
        """Normalize Russian text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Normalize ё to е
        text = re.sub(r'ё', 'е', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences"""
        if self.razdel:
            try:
                sentences = list(self.razdel.sentenize(text))
                return [sent.text for sent in sentences]
            except Exception as e:
                logger.warning(f"Razdel segmentation failed: {e}")
        
        # Fallback to simple splitting
        return [s.strip() for s in text.split('.') if s.strip()]
    
    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Chunk Russian text preserving sentence boundaries"""
        sentences = self.segment_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 10]
