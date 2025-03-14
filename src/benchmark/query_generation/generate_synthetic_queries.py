import os

### Imports LLMs Wrappers
from kotaemon.llms import (
    LCGeminiChat,
    LCOllamaChat,
)
import os

from benchmark.utils.custom_file_reader import SimplePDFReader, PdfExtractionReader
from benchmark.utils.utils import read_args, load_config
from query_gen_fct import (
    load_corpus,
    generate_qa_embedding_pairs,
)

def main():

    # Load config
    args = read_args()
    config = load_config(args.config_path)

    pdf_folder_path = str(config.get("source_folder"))  # Path to PDFs
    chunk_size = config.get("chunk_size", 1024)  # Chunk size for splitting text
    chunk_overlap = config.get("chunk_overlap", 0)  # Overlap between chunks
    output_folder = config.get("output_folder", "./outputs")
    output_filename = config.get("output_filename", "qa_pairs.json")
    verbose = bool(config.get("verbose", False))
    on_failure = config.get("on_failure", "continue")
    num_questions_per_chunk = config.get("num_questions_per_chunk", 1)
    num_files_limit = config.get("num_files_limit", -1)  # Max number of PDFs to process
    num_files_limit = config.get("num_files_limit", -1)
    filters = config.get("filters", ["syntax"])

    if num_files_limit == -1:
        num_files_limit = int(
            sum(
                1
                for file in os.listdir(pdf_folder_path)
                if file.lower().endswith(".pdf")
            )
        )

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    ### Set up an LLM model for question generations
    if config.get("llm_provider") == "ollama":
        llm_name = config.get("llm_model", "llama3.2:1b")  # LLM model name
        llm = LCOllamaChat(model=llm_name, num_ctx=8192)
    elif config.get("llm_provider") == "gemini":
        llm_name = config.get("llm_model")  # LLM model name
        api_key = config.get("api_key", None)
        api_key = config.get("api_key", None)  # Ensure it defaults to None
        if api_key is None:
            raise ValueError("API key is required for Gemini LLM provider.")
        llm = LCGeminiChat(api_key, model_name=llm_name)

    # Initialize a Custom PDF File Reader
    pdf_reader = PdfExtractionReader() # SimplePDFReader()  # or PDFThumbnailReader()

    list_of_nodes = load_corpus(
        pdf_folder_path,
        num_files_limit,
        pdf_reader,
        chunk_size,
        chunk_overlap,
        verbose=True,
    )

    generate_qa_embedding_pairs(
        llm=llm,
        nodes=list_of_nodes,
        qa_generate_prompt_tmpl=None,
        num_questions_per_chunk=num_questions_per_chunk,
        retry_limit=3,
        on_failure=on_failure,  # options are "fail" or "continue"
        save_every=10,
        verbose=verbose,
        output_path=output_path,
        query_filtering_fcts=filters,
    )


if __name__ == "__main__":
    main()
