import ast
import re
NULL_STRINGS = {
    "null",
    "none",
    "n/a",
    "na",
    "unknown",
    "unspecified",
    "",
    "N/A",
    "None",
}

QUOTE_CHARS = "«»“”\"'"

def canonicalize(value, enum_cls, allow_list=False):
    """
    Converts raw LLM output into canonical enum values.
    """
    tokens = normalize_to_set(value)

    if not tokens:
        return [] if allow_list else None

    enum_lookup = {
        clean_token(e.value): e.value
        for e in enum_cls
    }

    resolved = [
        enum_lookup[t]
        for t in tokens
        if t in enum_lookup
    ]

    if allow_list:
        return resolved

    return resolved[0] if resolved else None

def clean_token(s: str) -> str:
    """
    Canonicalizes a single token so semantic matches succeed.
    """
    s = s.strip()

    # Remove enclosing quotes / guillemets repeatedly
    while len(s) >= 2 and s[0] in QUOTE_CHARS and s[-1] in QUOTE_CHARS:
        s = s[1:-1].strip()

    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    return s.lower()


def normalize_to_set(x):
    """
    Canonical normalization for geography fields.
    Handles:
    - None
    - null-like strings
    - lists
    - stringified lists
    - decorative quotes (« » “ ”)
    """

    if x is None:
        return set()

    # List / tuple / set
    if isinstance(x, (list, tuple, set)):
        return {
            clean_token(str(v))
            for v in x
            if v is not None and clean_token(str(v)) not in NULL_STRINGS
        }

    # String
    if isinstance(x, str):
        raw = x.strip()

        # Clean first (handles « » etc.)
        cleaned = clean_token(raw)

        # Null-like
        if cleaned in NULL_STRINGS:
            return set()

        # Stringified list
        if cleaned.startswith("[") and cleaned.endswith("]"):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, (list, tuple)):
                    return {
                        clean_token(str(v))
                        for v in parsed
                        if clean_token(str(v)) not in NULL_STRINGS
                    }
            except Exception:
                return set()

        # Single value
        return {cleaned}

    return set()
