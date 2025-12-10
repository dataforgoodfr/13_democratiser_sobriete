import os
import pymupdf4llm


def get_markdown_pymupdf(
    pdf_path: str,
    bool_write_images: bool = False,
    bool_embed_images: bool = False,
) -> str:
    """
    Extract the content from a PDF file using pymupdf4llm.
    Args:
        pdf_path (str): The path to the PDF file.
        bool_write_images (bool): Whether to write images to disk.
        bool_embed_images (bool): Whether to embed images in the markdown.
    Returns:
        list[dict]: The extracted content.
    """
    # Get the name of the file from the path, without the extension
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract the content from the PDF
    md_text = pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=True,
        write_images=bool_write_images,
        embed_images=bool_embed_images,
        image_path=os.path.join("images_pdf", file_name),
        show_progress=False,
    )
    
    return md_text
