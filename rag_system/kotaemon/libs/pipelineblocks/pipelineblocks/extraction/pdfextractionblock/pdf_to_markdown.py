from pipelineblocks.extraction.pdfextractionblock.base import BasePdfExtractionBlock
from pipelineblocks.extraction.pdf_utils.pymu import get_pymupdf4llm

class PdfExtractionToMarkdownBlock(BasePdfExtractionBlock):

    """A Pdf extraction block, that convert a long pdf into chunks or a long text.
        Different methods could be implemented :
            - 'split_by_page' (by defaut), return a list of markdown, one by page of the original pdf
            - 'group_all' return the entire doc in a long markdown format
            
        TODO implement other methods like 'split_by_chunks' with the length of each chunk, or 'split_by_parts' with a logic split by part... """

    def run(self, pdf_path, method='split_by_page'):

        text_md_list = get_pymupdf4llm(pdf_path)

        if method == 'split_by_page':

            return text_md_list
        
        elif method == 'group_all':

            return "\n".join([c["text"] for c in text_md_list])