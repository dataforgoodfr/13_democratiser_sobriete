import re


def is_toc_line(line: str) -> bool:
    """For table of contents"""
    # 1. Détection des points de suite (le classique des sommaires)
    if "...." in line or " . . " in line:
        return True

    # 2. Détection de numéro de page en fin de ligne (ex: "Résultats ....... 45")
    if re.search(r"\s+\d+$", line.strip()):
        # Si la ligne est courte et finit par un chiffre, c'est suspect
        return True

    return False


def build_regex_for_section(keywords: str | list[str]):
    if isinstance(keywords, list):
        keywords = "|".join(keywords)
    pattern = rf"(?i)^(?:\d+\.?\s*)?(?:{keywords}).*"
    return pattern


patterns = {
    "abstract": build_regex_for_section(["abstract", "summary"]),
    "introduction": build_regex_for_section(["introduction", "background"]),
    "methods": build_regex_for_section(
        ["methods", "methodology", "materials? and methods?", "experimental design", "approach"]
    ),
    "results": build_regex_for_section(["results", "findings", "outcomes?"]),
    "discussion": build_regex_for_section(["discussion", "conclusions?", "implications", "limitations"]),
    "acknowledgements": build_regex_for_section(
        ["acknowledgements?", "acknowledgments?", "thanks", "funding"]
    ),
    "references": build_regex_for_section(
        ["references", "bibliography", "bibliographic references", "cited works", "sources", "literature cited"]
    ),
}


def extract_sections(lines: list[str], patterns: dict[str, str] = patterns) -> dict[str, str]:
    sections = {k: "" for k in patterns.keys()}
    sections["preamble"] = ""  # Tout ce qui est avant l'abstract

    current_section = "preamble"
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Vérification si la ligne est un titre de section
        # On limite la taille de la ligne pour éviter de confondre une phrase avec un titre
        found_new_section = False
        if len(clean_line) < 100 and not is_toc_line(clean_line):
            for section_name, pattern in patterns.items():
                if re.match(pattern, clean_line):
                    current_section = section_name
                    found_new_section = True
                    break

        if not found_new_section:
            sections[current_section] += line + "\n"

    return sections
