import os

import pymupdf4llm


def get_pymupdf4llm(
    pdf_path: str,
    bool_write_images: bool = False,
    bool_embed_images: bool = False,
) -> list[dict]:
    # Get the name of the file from the path, without the extension
    file_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract the content from the PDF
    content_md = pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=True,
        write_images=bool_write_images,
        embed_images=bool_embed_images,
        image_path=os.path.join("images_pdf", file_name),
        show_progress=False,
    )

    return content_md
