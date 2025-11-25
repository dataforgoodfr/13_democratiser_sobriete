# this is not a test as usually understood in python as there are no comparisons, but
# rather scripts to check the quality of the prompts generatated

from pathlib import Path

from library.pdfextraction.llm.utils import open_file
from library.pdfextraction.llm.prompts import main_parts_prompt


txt_path =  Path(__file__).parent / "1-s2.0-S0094119008001095-main.md"

def main():
    txt = open_file(txt_path)
    prompt = main_parts_prompt(txt)

    print(prompt)

    with open(str(txt_path).replace(".md", "") + "_main_parts_prompt.md", "w") as f:
        f.write(prompt)


if __name__ == "__main__":
    main()
