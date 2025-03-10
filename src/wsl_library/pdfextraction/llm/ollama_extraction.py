import argparse

from ollama import chat
from pydantic import BaseModel

from wsl_library.pdfextraction import TAXS, OLLAMA_MODELS
from wsl_library.pdfextraction.llm.utils import open_file, ollama_available
from wsl_library.pdfextraction.llm.prompts import basic_prompt, main_parts_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Extract OLLAMA from a paper")
    parser.add_argument("--text-path", type=str, help="Path to the text file")
    parser.add_argument(
        "--model", type=str, choices=OLLAMA_MODELS, help="Model to use for extraction"
    )
    parser.add_argument(
        "--taxonomy",
        type=str,
        choices=list(TAXS.keys()),
        default="PaperTaxonomy",
        help="Taxonomy to use for extraction",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["basic", "main_parts"],
        default=None,
        help="Type of prompt to use for extraction"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Name of the json where the extracted information will be stored"
    )

    return parser.parse_args()


def extract_ollama_from_paper(
    prompt: str, model: str, tax: BaseModel
) -> BaseModel:
    """
    Extract a taxonomy from a given prompt using the OLLAMA model.
    Args:
        prompt (str): The prompt to use for the extraction.
        model (str): The model to use for the extraction.
        tax (BaseModel): The taxonomy to use for the extraction.
    Returns:
        BaseModel: The extracted taxonomy (same type as the given taxonomy).
    """
    
    if not ollama_available(model):
        raise AttributeError(f"{model} is not among the local ollama models.")

    response = chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
        format=tax.model_json_schema()
    )

    paper = tax.model_validate_json(response.message.content)

    return paper


def main():
    args = parse_args()

    # check if the given taxonomy name is within the coded taxonomies
    if args.taxonomy:
        try:
            tax = TAXS[args.taxonomy]

        except KeyError as e:
            raise KeyError(f"Taxonomy {args.taxonomy} not found") from e

    # if no given taxonomies, use the default one   
    else:
        tax = tax = TAXS["PaperTaxonomy"]

    # open and process the text
    txt = open_file(args.text_path)

    if args.prompt == "main_parts":
        prompt = main_parts_prompt(txt)
    
    else:
        prompt = basic_prompt(txt)

    paper = extract_ollama_from_paper(prompt, args.model, tax)
    
    # print or save the output
    if not args.output_name:
        print(paper)
    
    # save the result with the given name
    else:
        output_path = args.output_name.split(".")[0] + ".json"
        with open(output_path, "w") as f:
            f.write(paper.model_dump_json())


if __name__ == "__main__":
    main()
