import re


def scientific_basic_prompt(text: str) -> str:
    """
    Create a prompt for the basic extraction of a scientific paper.
    Args:
        text (str): The text of the scientific paper.
    Returns:
        str: The prompt for the basic extraction.
    """

    prompt = f"""
    You are given a scientific paper. The first page corresponds to the where
    the title, authors, and abstract are located. The rest of the paper is
    divided into sections. Each section has a title and a body. The body of the
    section may contain text, figures, tables, and equations.
    You are tasked with extracting information from the paper.
    Here is the paper:
    {text}
    """

    return prompt


def scientific_system_prompt(article_content: str, openalex_metadata, paper_taxonomy) -> str:
    prompt = f"""
    You are an expert AI assistant tasked with extracting a structured taxonomy from a research paper.
    You are provided with: (1) authoritative metadata about the paper (OpenAlex data), (2) the full text of the paper, 
    and (3) the PaperTaxonomy schema defining the output format. Use the OpenAlex metadata for all fields it provides 
    (e.g. title, authors, venue, year, and any OpenAlex topics); these values are correct and should be used verbatim 
    without recalculation. For any fields in the PaperTaxonomy schema not already filled by OpenAlex data, 
    carefully infer them from the paper’s content. Rely only on the information in the paper itself 
    and the given metadata – do not use any external knowledge or ontologies. Focus on the paper’s key domains, 
    fields, and specific topics: identify the major areas of research and their subcategories discussed in the paper. 
    Organize these in a hierarchical manner if the schema requires (from broad domains to fine-grained topics), 
    ensuring each level reflects the paper’s content. Keep the taxonomy concise and relevant – include only 
    significant topics that are central to the paper. Ignore extraneous details (e.g. unrelated data, 
    bibliographic references, or minor tangents). When generating the output, strictly follow the PaperTaxonomy 
    schema structure. Provide the final answer as a JSON object that exactly matches the schema, with correct keys 
    and data types for each field. Do not include any explanations or any content outside the JSON. 
    Your output should be a valid, parseable instance of PaperTaxonomy, accurately capturing the paper’s taxonomy 
    while respecting all the guidelines above.
    
    :param article_content:
    {article_content}
    
    :param openalex_metadata:
    {openalex_metadata}
    """
    return prompt


def scientific_main_parts_prompt(text: str, output_format: dict | None = None) -> str:
    """
    Create a prompt for the extraction of the main parts of a scientific paper.
    A regular expression divides the text into parts, and the prompt is created
    using the parts that contain the introduction, results, and conclusion.
    Args:
        text (str): The text of the scientific paper.
    Returns:
        str: The prompt for the extraction of the main parts.
    """

    # extract the main sections from the pdf
    pattern_parts = r"(\*\*\d+\.(?P<title>[^*]+)\*\*[\s\S]+?)(?=\n\*\*|$)"
    sections = re.findall(pattern_parts, text)

    # find the intro section and the cover page coming before
    title_intro = [title for _, title in sections if "intro" in title.lower()][0]
    cover_page = text[:text.find(title_intro)]

    # find the conclusion and results part as well
    txt = "\n".join([
        t for t, title in sections if any(
            [m in title.lower() for m in ["results", "conclusion", "intro"]]
        )
    ])

    prompt = f"""
    You are given parts from a scientific paper, you are tasked with
    extracting information from these parts.

    Here is the text:
    {cover_page + txt}
    """

    return prompt
