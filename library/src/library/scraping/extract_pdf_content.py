import pathlib

# do not remove this import, it improves the results of pymupdf4llm
import pymupdf.layout  # noqa
import pymupdf4llm


def get_markdown_pymupdf(path: str,) -> str:
    md_text = pymupdf4llm.to_markdown(
        doc=path,
        header=False,
        footer=False,
    )
    return md_text


def save_markdown(text: str, output_path: str) -> None:
    pathlib.Path(output_path).write_bytes(text.encode())


def convert_pdf_to_markdown(
    pdf_path: str,
    output_md_path: str,
) -> None:
    md_text = get_markdown_pymupdf(pdf_path)
    save_markdown(md_text, output_md_path)
