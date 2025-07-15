from typing import List

from pipelineblocks.extraction.pdf_utils.pymu import get_pymupdf4llm
from pipelineblocks.extraction.pdfextractionblock.base import BasePdfExtractionBlock


class PdfExtractionToMarkdownBlock(BasePdfExtractionBlock):

    """A Pdf extraction block, that convert a long pdf into chunks or a long text.
    Different methods could be implemented :
        - 'split_by_page' (by default), return a list of markdown, one by page of the original pdf
        - 'group_all' return the entire doc in a long markdown format

    TODO implement other methods like 'split_by_chunks' with the length of each chunk,
    or 'split_by_parts' with a logic split by part..."""

    def word_splitter(self, source_text: str) -> List[str]:
        import re

        source_text = re.sub(r"\s+", " ", source_text)  # Replace multiple whitespaces
        return re.split(r"\s", source_text)  # Split by single whitespace

    def get_chunks_fixed_size_with_overlap(
        self, text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        text_words = self.word_splitter(text)
        chunks = []
        for i in range(0, len(text_words), chunk_size):
            chunk_words = text_words[max(i - overlap, 0) : i + chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)
        return chunks

    def run(
        self,
        pdf_path,
        method="split_by_page",
        chunk_size: int = 1024,
        chunk_overlap: int = 256,
    ):

        text_md_list = get_pymupdf4llm(pdf_path)

        if method == "split_by_page":

            return text_md_list

        else:

            text_md_all = "\n".join([c["text"] for c in text_md_list])

            if method == "group_all":
                return text_md_all

            elif method == "split_by_chunk":
                return self.get_chunks_fixed_size_with_overlap(
                    text_md_all, chunk_size=chunk_size, overlap=chunk_overlap
                )
