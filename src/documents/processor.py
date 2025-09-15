import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.documents.russian_nlp import RussianNLP

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.russian_nlp = RussianNLP()
    
    def process_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict:
        """Process document and extract text with metadata"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Extract text based on file type
            text_content = self._extract_text(file_path)
            
            # Process with Russian NLP
            normalized_text = self.russian_nlp.normalize_text(text_content)
            chunks = self.russian_nlp.chunk_text(normalized_text)
            
            result = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'raw_text': text_content,
                'normalized_text': normalized_text,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'metadata': metadata or {}
            }
            
            logger.info(f"✓ Processed {file_path.name}: {len(chunks)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise
    
    def _extract_text(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt':
            return self._read_text_file(file_path)
        else:
            # For now, just read as text - we'll enhance this later
            return self._read_text_file(file_path)
    
    def _read_text_file(self, file_path: Path) -> str:
        """Read plain text file with encoding detection"""
        encodings = ['utf-8', 'cp1251', 'koi8-r', 'iso-8859-5']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Could not decode text file: {file_path}")
        return ""
