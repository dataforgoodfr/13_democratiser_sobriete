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
    cover_page =  text[:text.find(title_intro)]

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
