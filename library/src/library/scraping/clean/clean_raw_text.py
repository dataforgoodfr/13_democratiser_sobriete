import re
import string


def merge_lines(lines):
    """
    Merge lines that belong to the same paragraph.
    An empty line doesn't necessarily indicate a new paragraph, so we must rely on poncutation.
    Hyphenated words at the end of a line are merged with the next line.
    Lines starting with a lowercase letter are merged with the previous line.
    """
    merged_lines = []
    buffer = ""
    previous_should_be_merged = False
    for line in lines:
        if not line.strip():
            continue  # skip empty lines

        clean_line = line.replace("\xa0", " ").replace("\uf0b7", "\u2022").strip()

        if buffer:
            if (
                line.startswith("\xa0")
                or previous_should_be_merged
                or re.match(r"^[a-z]", line)
            ):
                buffer += " " + clean_line
            elif buffer.endswith("-"):  # merge hyphenated word
                buffer = buffer[:-1] + clean_line
            else:
                merged_lines.append(buffer)
                buffer = clean_line
        else:
            buffer = clean_line

        previous_should_be_merged = line.endswith("\xa0")

    if buffer:
        merged_lines.append(buffer)
    return merged_lines


def is_garbage_line(line):
    line = line.strip()
    if not line:
        return True

    # 1. Supprime les lignes ne contenant que des chiffres (ex: numéros de page)
    # On autorise quelques espaces ou un point (ex: "12" ou "12.")
    if re.match(r"^[\d\s\.]+$", line):
        return True

    # 2. Supprime les lignes ne contenant que de la ponctuation (ex: "---", "...", "***")
    # On utilise string.punctuation pour couvrir tous les cas (!"#$%&'()*+, etc.)
    if all(c in string.punctuation or c.isspace() for c in line):
        return True

    # 3. Supprime les résidus de tableaux (ex: "1.2  0.5  0.8")
    # Si la ligne contient beaucoup de chiffres et peu de lettres, c'est du bruit.
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    if digits > 0 and letters < 3:  # Typique des lignes de données pures
        return True

    return False


def is_toc_line(line):
    """For table of contents"""
    # 1. Détection des points de suite (le classique des sommaires)
    if "...." in line or " . . " in line:
        return True

    # 2. Détection de numéro de page en fin de ligne (ex: "Résultats ....... 45")
    if re.search(r"\s+\d+$", line.strip()):
        # Si la ligne est courte et finit par un chiffre, c'est suspect
        return True

    return False


def clean_lines(lines):
    return [line for line in lines if not is_garbage_line(line) and not is_toc_line(line)]
