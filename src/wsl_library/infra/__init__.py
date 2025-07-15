import os

# Get the main directory of the project
# This is the directory where the project is stored
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Define the path to the PDF storage folder
# This is where the PDFs will be stored after being downloaded
PDF_STORAGE_FOLDER = os.path.join(PROJECT_DIR, "tests", "pdf")

# Define the path to the JSON storage folder
# This is where the JSON files will be stored after being downloaded
JSON_STORAGE_FOLDER = os.path.join(PROJECT_DIR, "tests", "pdfextraction", "llm", "outputs")

# Info on the databases to be used
# TODO
