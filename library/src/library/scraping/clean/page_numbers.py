import re


def remove_page_numbers(
    pages: list[str], page_numbers: list[tuple[int | None, str | None]]
) -> list[str]:
    """Remove page numbers from pages based on identified page_numbers list."""
    if page_numbers is None:
        return pages
    cleaned_pages = []
    for page, (pn, where) in zip(pages, page_numbers, strict=True):
        if pn is not None:
            if where == "start":
                # remove page number at the start
                page = re.sub(r"^(?:-?\s*)?(?:Page\s*)?\d{1,4}(?:\s*-)?", "", page).strip()
            elif where == "end":
                # remove page number at the end
                page = re.sub(r"(?:-?\s*)?(?:Page\s*)?\d{1,4}(?:\s*-)?$", "", page).strip()
        cleaned_pages.append(page)
    return cleaned_pages


def find_page_numbers(pages):
    page_nb = []
    start = 0
    end = 0
    for page in pages:
        page = page.strip()
        # find page number at the beginning or end of the page
        # match = re.search(r'^(?:Page\s*)?(\d{1,4})|(?:Page\s*)?(\d{1,4})$', page)
        match = re.search(
            r"^(?:-?\s*)?(?:Page\s*)?(\d{1,4})(?:\s*-)?|(?:-?\s*)?(?:Page\s*)?(\d{1,4})(?:\s*-)?$",
            page,
        )
        if match:
            if match.group(1):
                page_number = int(match.group(1))
                where = "start"
                start += 1
            else:
                page_number = int(match.group(2))
                where = "end"
                end += 1
            page_nb.append((page_number, where))
        else:
            page_nb.append((None, None))

    page_nb = keep_consistent(page_nb)

    pages_before = 0  # pages without a number before numbering starts
    pages_after = 0  # pages without a number after numbering ends
    numbered_pages = []
    for pn, _ in page_nb:
        if pn is None:
            if numbered_pages:
                pages_after += 1
            else:
                pages_before += 1
        else:
            pages_after = 0
            numbered_pages.append(pn)

    if len(numbered_pages) < 2 or len(numbered_pages) / len(pages) < 0.2:
        return None

    page_diff = max(numbered_pages) - min(numbered_pages)
    is_likely_correct = abs(page_diff - len(pages)) <= 1 + pages_before + pages_after

    if is_likely_correct:
        min_page = min(numbered_pages) - pages_before
        max_page = max(numbered_pages) + pages_after
        pages_range = list(range(min_page, max_page + 1))
        # fill in missing page numbers
        current_index = 0
        for i in range(len(page_nb)):
            pn, where = page_nb[i]
            if pn is None:
                page_nb[i] = (pages_range[current_index], "filled")
            current_index += 1
        return page_nb

    return None


def keep_consistent(
    page_numbers: list[tuple[int | None, str | None]],
) -> list[tuple[int | None, str | None]]:
    """
    Expects a list of (page_number, where) tuples, where "where" is either 'start' or 'end' depending on the position of the number on the page.
    Returns a list of the same length, with inconsistent page numbers set to None.
    Consistency is determined by checking if page numbers increment by the number of items from the last valid page number.
    Since page number is either always at the start or always at the end, we first determine which 'where' is dominant and set others to None.
    """
    # Count occurrences of 'start' and 'end'
    where_counts = {"start": 0, "end": 0}
    for pn, where in page_numbers:
        if where in where_counts and pn is not None:
            where_counts[where] += 1

    # Determine dominant 'where'
    if where_counts["start"] > where_counts["end"]:
        dominant_where = "start"
    elif where_counts["end"] > where_counts["start"]:
        dominant_where = "end"
    else:
        dominant_where = None

    # If a dominant 'where' is found, set others to None
    if dominant_where is not None:
        for i, (_, where) in enumerate(page_numbers):
            if where != dominant_where:
                page_numbers[i] = (None, None)

    last_valid = None
    for i, (pn, _) in enumerate(page_numbers):
        if pn is not None:
            if last_valid is None:
                last_valid = (i, pn)
                continue
            expected_pn = last_valid[1] + (i - last_valid[0])
            if pn == expected_pn:
                last_valid = (i, pn)
            else:
                page_numbers[i] = (None, None)
                page_numbers[last_valid[0]] = (None, None)
                last_valid = None

    return page_numbers
