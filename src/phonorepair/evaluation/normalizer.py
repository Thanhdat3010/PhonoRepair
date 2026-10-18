"""Symmetric text normalization v1 for Interspeech benchmarks.

Matches Proposal Section 11.1:
1. Unicode NFKC normalization and lowercase.
2. Replace punctuation with space, preserving apostrophes within words (e.g. don't).
3. Collapse multiple whitespace characters.
4. Single digits 0-9 mapped to fixed English words (0->zero, 1->one, etc.).
"""

import re
import unicodedata

# Single-digit mapping table
DIGIT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def normalize_text_v1(text: str) -> str:
    """Normalize transcript text symmetrically for baseline and repair."""
    if not text:
        return ""

    # 1. Unicode NFKC & lowercase
    normalized = unicodedata.normalize("NFKC", text).lower()

    # 2. Preserve apostrophes inside words (e.g. can't, it's), replace other punctuation
    # Replace any punctuation except apostrophe
    normalized = re.sub(r"[^\w\s']", " ", normalized)

    # Remove apostrophes that are not surrounded by alphanumeric characters
    normalized = re.sub(r"(?<!\w)'|'(?!\w)", " ", normalized)

    # 3. Collapse whitespace
    tokens = normalized.strip().split()

    # 4. Map single digits to words
    mapped_tokens = [DIGIT_MAP.get(token, token) for token in tokens]

    return " ".join(mapped_tokens)
