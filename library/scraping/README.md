# Scraping

This part of the project aims at transforming the metadata obtained from the prescreening phase into a usable text dataset ready for ingestion into a vector DB and for policy analysis.

## Obtaining usable PDFs

The prescreening phase gave us ~2.5M publications from OpenAlex, with ~1.7M of them open access.
Among those, ~1.3M have a direct PDF URL, while the rest only have web URLs, which some of the time contain the full text and most of the time links to a landing page with a button to download the PDF.

Out of simplicity, we chose to focus on publications with a direct PDF URL.
Some of those link to actual PDF files, while other link to web pages loading PDF with javascript, or web pages anti-scraping measures.

Trying to obtain the full text from all those formats would take an unreasonable amount of effort for this project.
Thus, again out of simplicty, we only processed publications with real PDF URL, yielding ~690k PDF files that we downloaded (which wasn't mandatory and takes about 2TB).

We divide those into 6 batches of 100k files and 1 batch (batch_7) of 90k files.
This batch structure is preserved for the following steps.

The script used to do so is `download_all_pdfs.py`, with helper functions in `src/scraping/download_pdf`.

## Extracting text from PDFs

PDF text extraction is more challenging that it sounds, especially at scale.
The goal is to convert each PDF into an equivalent text file, usually in markdown format.
Popular libraries like docling handle it very well, but they are too expensive to run for our scale and means : quick tests showed, after extrapolation, that converting the 690k PDF files into markdown would require ~8500 H100-hours - almost a full year.

A much faster library with good results is `pymupdf4llm`.
However, we encountered major hurdles in the form of memory issues, with OOM errors killing our jobs quickly.
We could not overcome them despite several days of debugging.

- Disabling OCR didn't work
- Reducing the memory overhead of the main process didn't work.
- Processing pages one by one or by larger chunks didn't work.
- Limiting the maximum memory used by a single worker didn't work.

Thus, we switched to a worse-quality but simple and very fast method: extracting raw text from PDFs without layout analysis.

The script to do so (with md or raw text option) is `extract_text_from_pdfs.py`, with helper functions in `src/scraping/extract_pdf_content.py`.

This creates a txt file for each pdf file in a given folder.
We then gather these texts into a parquet file using `save_txt_as_parquet`.

Raw text is low-quality: sentences are interrupted by random line breaks virtually indistinguishible from real line breaks that we want to keep ; headers, footers and page numbers can appear in the middle of a paragraph if it spanned two pages ; and tables appear completely destructured, with often one cell value per line in a manner that makes it impossible to confidently reconstruct.

Raw text must be cleaned : headers, footers, pages numbers, tables and other garbage lines should be removed. Cleaning code is available in `src/scraping/clean` and gather in `src/scraping/clean/cleaning_pipeline.py`.

## Extracting sections

It was decided to only keep the results and discussion/conclusion sections of each article, to reduce scale and thus decrease cost and latency, and having a higher signal-to-noise ratio for policy analysis.

The markdown format makes it easy using hashtags (although PDF-to-md tools don't preserve title hierachy well), but since we ended up with raw text, we used a regex-based method. The code is in `src/scraping/extract_sections.py`.

The script that takes the raw text, cleans it and extract sections is `extract_sections_from_raw_text.py`.
It saves for each batch the results as a dataframe in a `processed_text.parquet` file.

We then created a "final" parquet file with only results and conclusions from all batches : `results_conclusions_585k_2025-01-02.parquet`.
It contains 585k lines out of 690k inital documents, as not all of them contain one of those sections.

## Quick start

Install uv:

```
https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you plan to use pymupdf4llm OCR (not recommended), [install tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) :

```
sudo apt install tesseract-ocr
```

Create the venv using group pdfscraping:

```
uv sync --group pdfscraping
```

Group webscraping is used by old code that used selenium to obtain more PDFs.

Run the script of your choice using the venv :

```
uv run python myscript.py [cli args]
```

Read the text above to undertand which script to run.
