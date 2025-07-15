def generic_extraction_prompt_entire_doc(text: str, language: str = "English") -> str:
    """
    Create a prompt for the basic extraction of any kind of document.
    Args:
        text (str): The text of the document.
    Returns:
        str: The prompt for the basic extraction.
    """

    prompt = f"""
    You are a very attentive and meticulous reader. The first page of the document corresponds
    to the where the title, perhaps the authors. The rest of the paper is
    divided into sections or chapters. Each section has a title and a body. The body of the
    section may contain text, figures, tables, others content...
    You are tasked with extracting information from the document.
    Please extract the information in the language of the original text.
    This language is the {language}.
    Don't force yourself to fill in everything if it doesn't seem relevant.
    You can leave some fields blank if you haven't found the information you're looking for.
    For example, if you have to fill in certain fields in the form of a list,
    you can leave a list completely empty if you don't find any relevant information.
    Here is the document :
    {text}
    """

    return prompt


def generic_extraction_prompt_chunk(text: str, language: str = "English") -> str:
    """
    Create a prompt for the basic extraction of a chunk extract from a textual document.
    A chunk is a 'random part' of the document, divided by size, or divided by page.
    Args:
        text (str): The text of the document.
    Returns:
        str: The prompt for the basic extraction.
    """

    prompt = f"""
    You are a very attentive and meticulous reader.
    The chunk here is extracted from a more complete document.
    You are tasked with extracting information from this chunk.
    Please extract the information in the language of the original text.
    This language is the {language}.
    Don't force yourself to fill in everything if it doesn't seem relevant.
    You can leave some fields blank if you haven't found the information you're looking for.
    For example, if you have to fill in certain fields in the form of a list,
    you can leave a list completely empty if you don't find any relevant information.
    Here is the chunk :
    {text}
    """

    return prompt
