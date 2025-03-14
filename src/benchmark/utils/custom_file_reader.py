import os
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple

import nltk
from pypdf import PdfReader

from llama_index.core.readers.base import BaseReader
from llama_index.core import Document

from wsl_library.pdfextraction.pdf.pymu import get_pymupdf4llm

def dehyphenate(lines: List[str], line_no: int) -> List[str]:
    if line_no + 1 >= len(lines):  
        return lines  # Avoid IndexError

    word_suffix, _, rest = lines[line_no + 1].partition(" ")
    lines[line_no] = lines[line_no][:-1] + word_suffix
    lines[line_no + 1] = rest
    return lines


def remove_hyphens(text: str) -> str:
    """
    Remove hyphenation across line breaks while preserving meaningful dashes.
    """
    lines = [line.rstrip() for line in text.split("\n")]
    line_numbers = [i for i, line in enumerate(lines[:-1]) if line.endswith("-")]

    for line_no in line_numbers:
        lines = dehyphenate(lines, line_no)

    return "\n".join(lines)

def remove_page_numbers(text: str) -> str:
    """
    Remove common page number patterns from extracted PDF text.
    """
    patterns = [
        r"^\s*\d+\s*$",  # Standalone numbers
        r"\bPage\s*\d+\b",  # "Page X"
        r"\bPage\s*\d+\s*of\s*\d+\b",  # "Page X of Y"
        r"^\s*\f\s*$",  # Form feed characters (page breaks)
    ]
    
    regex = re.compile("|".join(patterns), re.MULTILINE)
    return regex.sub("", text)



def handling_authors(text):
    """Preserve periods inside parentheses and replace ' al.' with ' al[DOT]' before tokenization."""

    # Preserve periods inside parentheses (e.g., "(Smith et al., 2020)")
    text = re.sub(r"\(([^)]+)\)", lambda m: m.group(0).replace(".", "[DOT]"), text)

    # Replace " al." with " al[DOT]" to prevent incorrect sentence splitting
    text = re.sub(r"\b(al)\.", r"\1[DOT]", text)

    return text


def get_sentences_from_text(text: str) -> List[str]:
    """Split text into sentences while preserving citations."""
    text = handling_authors(text)
    sentences = nltk.sent_tokenize(text)
    return list(map(lambda s: s.replace("[DOT]", ".").replace(" -", "-"), sentences))



def preprocess_text(text):
    """Preprocess text by removing hyphens and page numbers."""
    text = remove_page_numbers(text)
    text = remove_hyphens(text)
    sentences = get_sentences_from_text(text)
    sentences = [s.replace("\n", " ") for s in sentences]
    text = "\n".join(sentences)
    return text


def extract_doi(reader: PdfReader) -> Optional[str]:
    """Extract DOI from the first few pages of the PDF."""
    doi_pattern = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)

    for page in reader.pages:
        text = page.extract_text() or ""
        match = doi_pattern.search(text)
        if match:
            return match.group()
    return None



def extract_text_from_pdf(reader, preprocess_text_function=None) -> str:
    full_text = ""
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # Only process non-empty text
            if preprocess_text_function:
                full_text += preprocess_text_function(text) + "\n"
            else:
                full_text += text + "\n"

    return full_text.strip()


def convert_pdf_to_text(pdf_filename, preprocess_text_function=None, doi=False):

    size_of_pdf = os.path.getsize(pdf_filename)

    if size_of_pdf < 3000:
        return None, None

    with open(pdf_filename, "rb") as f:
        first_bytes = f.read(10)

    if b"PDF" not in first_bytes:
        return None, None

    try:
        reader = PdfReader(pdf_filename)
        text = extract_text_from_pdf(reader, preprocess_text_function)
        if doi:
            doi = extract_doi(reader)
            return text, doi
        return text, None
    except Exception as e:
        print(pdf_filename, e)
        return None, None


def pdfs_processing(file_path: str, preprocess_text_function=None) -> Tuple[Optional[str], dict]:
    """Process PDF file and extract metadata."""
    
    metadata = {
        "doi": None,
        "title": None,
        "publication_date": None,
        "publication_year": None,
        "language": None,
        "type": None,
    }

    metadata_file_path = Path(file_path).with_suffix(".json")

    try:
        if metadata_file_path.exists():
            text, _ = convert_pdf_to_text(file_path, preprocess_text_function, doi=False)

            with open(metadata_file_path, "r", encoding="utf-8") as f:
                json_metadata = json.load(f)

            metadata.update({k: v for k, v in json_metadata.items() if v is not None})
        
        else:
            text, doi = convert_pdf_to_text(file_path, preprocess_text_function, doi=True)
            metadata["doi"] = doi

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        text = None

    return text, metadata


class SimplePDFReader(BaseReader):
    def load_data(self, file, extra_info=None):

        text, metadata = pdfs_processing(file, preprocess_text_function=preprocess_text)
        if text is None:
            return []

        return [Document(text=text, metadata=metadata)]

class PdfExtractionReader(BaseReader):
    def load_data(self, file: str, extra_info=None):
        """Load and extract structured data from a PDF."""
        _, metadata = pdfs_processing(file, preprocess_text_function=None)
        content_md = get_pymupdf4llm(pdf_path=file, page_chunks=True)

        if not content_md:
            return []

        # md_metadata = content_md[0].get("metadata", {})
        # metadata.update({k: v for k, v in md_metadata.items() if v is not None})

        text = "\n".join(c["text"] for c in content_md)
        return [Document(text=text, metadata=metadata)]