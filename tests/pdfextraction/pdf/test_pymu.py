# this is not a test as usually understood in python as there are no comparisons, but
# rather a script to check if the code act as intented.

import os
from pathlib import Path

from wsl_library.pdfextraction.pdf.pymu import get_pymupdf4llm


HERE_PATH = Path(__file__).parent

def main():
    files = [f for f in os.listdir(HERE_PATH) if f.endswith(".pdf")]
    for f in files:
        md = get_pymupdf4llm(HERE_PATH / f)
        text = "\n".join([c["text"] for c in md])

        output_name = HERE_PATH / f.replace(".pdf", ".md")
        
        with open(output_name, "w") as f:
            print(f"writing at {output_name}")
            f.write(text)

if __name__ == "__main__":
    main()
