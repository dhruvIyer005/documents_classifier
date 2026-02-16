"""text_processor.py - Robust PDF extraction and chunking."""

import logging
import re
from collections import Counter
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
import torch

from config import config

CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP
MAX_LENGTH = config.MAX_LENGTH
MIN_TEXT_LENGTH = config.MIN_TEXT_LENGTH
MAX_PAGES = config.MAX_PAGES

logger = logging.getLogger(__name__)


class TextProcessor:
    """Process PDFs: extract, clean, and chunk for transformers."""

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP
        self.max_length = MAX_LENGTH

    def extract_text(self, pdf_path: str) -> Optional[str]:
        """
        Extract cleaned text from a PDF.
        - Removes headers/footers/page numbers/repeated lines
        - Stops at the first 3000 words
        - Handles empty/scanned PDFs gracefully
        """
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Error opening {pdf_path}: {e}")
            return None

        try:
            page_lines = []
            num_pages = min(len(doc), MAX_PAGES)

            # Collect raw lines per page
            for idx in range(num_pages):
                page = doc[idx]
                text = page.get_text("text") or ""
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                page_lines.append(lines)

            doc.close()

            # If nothing extracted, likely scanned or empty
            if not any(page_lines):
                logger.warning(f"No extractable text found in {pdf_path}")
                return None

            # Count line repetition across pages to drop headers/footers
            flat_lines = [ln for lines in page_lines for ln in lines]
            counts = Counter(flat_lines)
            repeated = {ln for ln, c in counts.items() if c > 1 and len(ln) <= 120}

            def is_page_number(line: str) -> bool:
                return bool(re.match(r"^\s*\d+\s*$", line))

            cleaned_pages: List[str] = []
            for lines in page_lines:
                keep: List[str] = []
                stop_refs = False
                for ln in lines:
                    low = ln.lower()
                    if is_page_number(ln):
                        continue
                    if ln in repeated:
                        continue
                    if "references" == low or low.startswith("references " ):
                        stop_refs = True
                        continue
                    if stop_refs:
                        continue
                    keep.append(ln)

                # Drop consecutive duplicates inside the page
                dedup_keep = []
                for ln in keep:
                    if not dedup_keep or dedup_keep[-1] != ln:
                        dedup_keep.append(ln)
                cleaned_pages.append(" ".join(dedup_keep))

            full_text = " ".join(cleaned_pages)
            words = full_text.split()
            if not words:
                logger.warning(f"No usable text after cleaning in {pdf_path}")
                return None

            # Limit to first 3000 words
            limited_text = " ".join(words[:3000])

            if len(limited_text) < MIN_TEXT_LENGTH:
                logger.warning(f"Text too short after cleaning: {len(limited_text)} chars < {MIN_TEXT_LENGTH}")
                return None

            return limited_text

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def chunk_text(self, text: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Split text into 400-450 token chunks with 50-token overlap."""
        if not text or len(text) < MIN_TEXT_LENGTH:
            return []

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 50:
            logger.warning(f"Too few tokens: {len(tokens)}")
            return []

        chunks = []
        stride = self.chunk_size - self.chunk_overlap

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + self.chunk_size]

            input_ids = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(input_ids)

            pad_length = self.max_length - len(input_ids)
            if pad_length > 0:
                input_ids += [self.tokenizer.pad_token_id] * pad_length
                attention_mask += [0] * pad_length

            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

            chunks.append(
                (
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                )
            )

        return chunks
