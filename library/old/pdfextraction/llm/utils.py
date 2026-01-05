import ollama


def open_file(text_path: str) -> str:
    """
    Open a file and return its content.
    Args:
        text_path (str): The path to the file.
    Returns:
        str: The content of the file
    """
    with open(text_path, "r", encoding="utf-8") as f:
        txt = f.read()
    
    return txt

def ollama_available(model_name: str) -> bool:
    """
    Check if the given model is available in the local ollama models.
    Args:
        model_name (str): The name of the model.
    Returns:
        bool: True if the model is available, False otherwise.
    """
    models = ollama.list().get("models", [])
    return any(model["model"] == model_name for model in models)
