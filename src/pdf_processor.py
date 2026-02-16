# pdf_processor.py
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF
from config import config

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PDFTextExtractor:
    """Extract FULL text from PDF files and chunk for processing"""
    
    def __init__(self):
        self.min_text_length = config.MIN_TEXT_LENGTH
        self.max_pages = config.MAX_PAGES
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.extract_n_chars = 10000  # Limit text to 10000 characters
        
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract ALL text from PDF file.
        Returns concatenated text from all pages.
        """
        try:
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return None
            
            # Extract using PyMuPDF
            text = self._extract_with_pymupdf(pdf_path)
            if text:
                return self._preprocess_text(text)
            
            logger.warning(f"Could not extract text from {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}", exc_info=True)
            return None
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Optional[str]:
        """Extract ALL text from all pages using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            # Extract text from ALL pages (or up to max_pages)
            num_pages = min(len(doc), self.max_pages)
            
            for page_num in range(num_pages):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"
            
            doc.close()
            logger.debug(f"PyMuPDF extracted {len(text)} chars from {pdf_path}")
            return text.strip()
            
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
            return None
    
    def chunk_text(self, text: str, tokenizer) -> List[Tuple[List[int], List[int]]]:
        """
        Chunk text into 400-450 token chunks with overlap.
        
        Args:
            text: Full document text
            tokenizer: Tokenizer to use for token counting
            
        Returns:
            List of (input_ids, attention_mask) tuples
        """
        if not text or len(text.strip()) < self.min_text_length:
            return []
        
        # Tokenize the entire text
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            if len(chunk_tokens) < 50:  # Skip very small chunks
                continue
            
            # Add special tokens
            input_ids = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)
            
            # Pad to max_length
            pad_length = config.MAX_LENGTH - len(input_ids)
            if pad_length > 0:
                input_ids += [tokenizer.pad_token_id] * pad_length
                attention_mask += [0] * pad_length
            
            # Truncate if longer than max_length
            input_ids = input_ids[:config.MAX_LENGTH]
            attention_mask = attention_mask[:config.MAX_LENGTH]
            
            chunks.append((input_ids, attention_mask))
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess extracted text for classification"""
        if not text or len(text.strip()) < self.min_text_length:
            return ""
        
        # Clean text
        text = ' '.join(text.split())  # Remove extra whitespace
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = text[:self.extract_n_chars]  # Limit length
        
        return text.strip()
    
    def extract_text_with_metadata(self, pdf_path: str) -> Dict:
        """Extract text and basic metadata"""
        text = self.extract_text(pdf_path)
        
        if not text:
            return {}
        
        # Get basic metadata
        try:
            with fitz.open(pdf_path) as doc:
                page_count = len(doc)
        except:
            page_count = 0
        
        return {
            'text': text,
            'page_count': page_count,
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path)
        }


class PDFDatasetBuilder:
    """Build dataset from PDF files for classification training"""
    
    def __init__(self):
        self.extractor = PDFTextExtractor()
        
    def process_pdfs_by_label(self, base_dir: str) -> pd.DataFrame:
        """
        Process PDFs organized by label directories:
        base_dir/
          ├── ieee/*.pdf
          ├── springer/*.pdf
          ├── acm/*.pdf
          ├── compliance/*.pdf
          └── legal/*.pdf
        """
        data = []
        base_path = Path(base_dir)
        
        logger.info(f"Processing PDFs from: {base_dir}")
        
        # Validate base directory
        if not base_path.exists():
            logger.error(f"Base directory not found: {base_dir}")
            return pd.DataFrame()
        
        for label in config.LABELS:
            label_dir = base_path / label
            
            if not label_dir.exists():
                logger.warning(f"Label directory '{label}' not found in {base_dir}")
                continue
            
            logger.info(f"Processing {label} documents...")
            pdf_files = list(label_dir.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDFs found in {label_dir}")
                continue
            
            for pdf_file in pdf_files:
                logger.debug(f"  Processing: {pdf_file.name}")
                
                # Extract text
                result = self.extractor.extract_text_with_metadata(str(pdf_file))
                
                if result and result.get('text'):
                    # Create data entry
                    entry = {
                        'text': result['text'],
                        'label': label,
                        'label_id': config.LABEL2ID[label],
                        'file_name': result['file_name'],
                        'file_path': str(pdf_file),
                        'page_count': result['page_count']
                    }
                    data.append(entry)
                    logger.info(f"    ✓ Extracted {len(result['text'])} chars")
                else:
                    logger.warning(f"    ✗ Failed to extract text")
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data)
            logger.info(f"Successfully processed {len(df)} PDFs")
            return df
        else:
            logger.warning("No PDFs were successfully processed")
            return pd.DataFrame()
    
    def process_single_pdf(self, pdf_path: str, label: Optional[str] = None) -> Dict:
        """Process a single PDF file for prediction"""
        result = self.extractor.extract_text_with_metadata(pdf_path)
        
        if result and result.get('text'):
            return {
                'text': result['text'],
                'label': label,
                'file_name': result['file_name'],
                'file_path': pdf_path,
                'page_count': result['page_count']
            }
        return {}
    
    def process_bulk_pdfs(self, pdf_paths: List[str]) -> pd.DataFrame:
        """Process multiple PDFs for bulk classification"""
        data = []
        
        print(f"Processing {len(pdf_paths)} PDFs for bulk classification...")
        
        for pdf_path in pdf_paths:
            result = self.extractor.extract_text_with_metadata(pdf_path)
            
            if result and result.get('text'):
                entry = {
                    'text': result['text'],
                    'file_name': result['file_name'],
                    'file_path': pdf_path,
                    'page_count': result['page_count'],
                    'label': None,  # To be predicted
                    'label_id': None
                }
                data.append(entry)
                print(f"  Processed: {result['file_name']}")
            else:
                print(f"  Failed: {os.path.basename(pdf_path)}")
        
        return pd.DataFrame(data)


# For backward compatibility
class PDFProcessor(PDFDatasetBuilder):
    """Legacy class name support"""
    pass


def extract_text_for_classification(pdf_path: str) -> str:
    """Simple function to extract text from PDF for classification"""
    extractor = PDFTextExtractor()
    return extractor.extract_text(pdf_path) or ""