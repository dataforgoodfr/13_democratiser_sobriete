import pathlib
import gc

# do not remove this import, it improves the results of pymupdf4llm
import pymupdf.layout  # noqa
import pymupdf4llm


def get_markdown_pymupdf(path: str, ocr: bool = False, max_pages_at_once: int = 20) -> tuple[str, bool]:
    """
    Extract markdown text from a PDF using pymupdf4llm, with OCR fallback.
    Returns tuple (markdown_text: str, used_ocr: bool)
    """

    with pymupdf.open(path) as doc:
        total_pages = len(doc)
        used_ocr = False
        all_texts = []
        chunk_size = max_pages_at_once or total_pages

        for start in range(0, total_pages, chunk_size):
            end = min(start + chunk_size, total_pages)
            page_range = list(range(start, end))
            
            text = pymupdf4llm.to_markdown(
                doc=doc,
                pages=page_range,
                header=False,
                footer=False,
                use_ocr=False,
                force_text=False
            )

            if ocr and needs_ocr(text):
                text = pymupdf4llm.to_markdown(
                    doc=doc,
                    pages=page_range,
                    header=False,
                    footer=False,
                    use_ocr=True,
                    force_text=False
                )
                used_ocr = True

            all_texts.append(text)

        md_text = "\n\n".join(all_texts)
        gc.collect()
        return md_text, used_ocr


def save_markdown(text: str, output_path: str) -> None:
    pathlib.Path(output_path).write_bytes(text.encode())


def convert_pdf_to_markdown(
    pdf_path: str,
    output_md_path: str,
) -> None:
    md_text = get_markdown_pymupdf(pdf_path)
    save_markdown(md_text, output_md_path)


def needs_ocr(extracted_text: str) -> bool:
    """Check if extracted_text is meaningful or if OCR is needed."""
    text_stripped = extracted_text.strip()
    word_count = len(text_stripped.split())
    char_count = len(text_stripped)
    
    # Use OCR fallback if:
    # - Very short output (< 200 chars suggests scanned/image-based)
    # - Low word count (< 50 words)
    # - Suspiciously low chars-to-words ratio (< 3.0 suggests garbled/sparse text)
    return (
        char_count < 200 or 
        word_count < 50 or 
        (word_count > 0 and char_count / word_count < 3.0)
    )