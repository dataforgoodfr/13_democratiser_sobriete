from library.scraping.clean.clean_raw_text import clean_lines, merge_lines
from library.scraping.clean.page_numbers import remove_page_numbers, find_page_numbers
from library.scraping.clean.headers_footers import (
    find_headers_footers,
    remove_headers_footers,
)


def normalize_whitespaces(text):
    return " ".join(text.split())


def cleaning_pipeline(text):
    pn, headers, footers = identify_headers_footers(text)

    pages = text.split(chr(12))
    pages = remove_page_numbers(pages, pn)
    pages = remove_headers_footers(pages, headers, footers)

    cleaned_text = "\n".join(pages)
    # cleaned_text = fix_text(cleaned_text)
    lines = cleaned_text.split("\n")
    merged_lines = merge_lines(lines)
    cleaned_lines = clean_lines(merged_lines)
    return "\n".join(cleaned_lines)


def identify_headers_footers(text: str):
    # 1. Split into pages removing \n
    pages = [normalize_whitespaces(s.replace("\n", " ")) for s in text.split(chr(12))]

    # 2. Identify page numbers
    page_numbers = find_page_numbers(pages)

    # 3. Remove identified page numbers
    if page_numbers is not None:
        pages = remove_page_numbers(pages, page_numbers)

    # 4. Identify headers and footers
    headers, footers = find_headers_footers(pages, min_words=2, min_occurrences=3)

    if page_numbers is None:
        # 5. Remove identified headers and footers
        if headers or footers:
            pages = remove_headers_footers(pages, headers, footers)

        # 6. Re-identify page-numbers (for when at the end of header or start of footer)
        page_numbers = find_page_numbers(pages)

    return page_numbers, headers, footers
