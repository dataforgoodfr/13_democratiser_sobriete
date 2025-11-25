import argparse
import json
import os
from pathlib import Path

import pymupdf4llm

from library.taxonomy.paper_taxonomy import OpenAlexPaper, PaperWithText

def get_args_parser():
    parser = argparse.ArgumentParser(description="Extract text from a PDF file.")
    parser.add_argument(
        "--pdf-path",
        type=str,
        help="Path to the PDF file to extract content from.",
    )
    parser.add_argument(
        "--write-images",
        type=bool,
        default=False,
        help="Write images to disk.",
    )
    parser.add_argument(
        "--embed-images",
        type=bool,
        default=False,
        help="Embed images in the markdown.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output file.",
    )
    return parser

# TODO : add wrapper function to abscract the extraction of the content regardless of the library used, and normalise the output
def get_pymupdf4llm(
    pdf_path: str,
    bool_write_images: bool = False,
    bool_embed_images: bool = False,
) -> list[dict]:
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
    content_md = pymupdf4llm.to_markdown(
        doc=pdf_path,
        page_chunks=True,
        write_images=bool_write_images,
        embed_images=bool_embed_images,
        image_path=os.path.join("images_pdf", file_name),
        show_progress=False,
    )
    
    return content_md

def extract_text(papers_list: list[OpenAlexPaper]) -> list[PaperWithText]:
    extracted_text_list = []
    for pix, paper in enumerate(papers_list):
        print(f"Extracting text from paper's PDF : {pix + 1}/{len(papers_list)}")
        if paper.pdf_path:
            pdf_content = get_pymupdf4llm(
                pdf_path=paper.pdf_path,
                bool_write_images=False,
                bool_embed_images=False,
            )
            full_text = "\n".join([c["text"] for c in pdf_content])
            
            # TODO : Test to fill only once the list when adjusting the PaperWithText model and fill missing fields with None
            extracted_text_list.append(
                PaperWithText(
                    openalex_paper  = paper,
                    extract_text    = full_text,
                    extrated_object = pdf_content,
                )
            )
        else:
            # TODO : Extract text from the metadata
            extracted_text_list.append(
                PaperWithText(
                    openalex_paper  = paper,
                    extract_text    = "some metadata",
                )
            )
    return extracted_text_list

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    content_md = get_pymupdf4llm(
        pdf_path=args.pdf_path,
        bool_write_images=args.write_images,
        bool_embed_images=args.embed_images,
    )
    if not args.pdf_path.endswith(".pdf"):
        raise ValueError("The file must be a PDF file.")        

    article_name = args.pdf_path.split("/")[-1].split(".pdf")[0]

    if args.output:
        print(len(content_md))
        # print(content_md[0]["text"])
        metadata = content_md[0]["metadata"]
        text = "\n".join([c["text"] for c in content_md])
        
        folder_name = Path(args.output.split(".")[0])
        os.makedirs(folder_name, exist_ok=True)
        text_name = folder_name / (article_name + ".md")
        json_name = folder_name / (article_name + ".json")

        with open(text_name, "w") as f:
            print(f"saving at {text_name}")
            f.write(text)

        with open(json_name, "w") as f:
            json.dump(metadata, f, indent=4)

    else:
        print("\n".join([c["text"] for c in content_md]))


if __name__ == "__main__":
    main()
