"""
trustboost/detector/pattern_detector.py
---------------------------------------
PURPOSE:
    A fast BACKUP detector for PII that always has a fixed shape
    (email, PAN, IFSC, Aadhaar, phone, IP address, card numbers, keys...).

    It does NOT replace the Gemma model. Gemma still handles everything that
    needs understanding (names, addresses, confidential projects). This file
    only catches fixed-format items the model might miss, and anything found
    by EITHER detector gets masked.

    It returns the same shape as the model: [{"text", "type", "reason"}].
    The main detector then finds every position of each value in the text.

SWITCH:
    Set TRUSTBOOST_PATTERNS=0 in your .env file to turn this backup off
    (useful to measure how much it helps).
"""

import re

# (entity type, compiled pattern). Order does not matter.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # PAN: 5 capital letters, 4 digits, 1 capital letter (e.g. ABCDE1234F)
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
    # IFSC: 4 letters, a zero, then 6 letters/digits (e.g. SBIN0001234)
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    # Aadhaar: 12 digits, first digit 2-9, optionally in groups of 4
    ("AADHAAR", re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")),
    # Indian mobile: optional +91, then 10 digits starting with 6-9
    ("PHONE", re.compile(r"(?<![\d+])(?:\+91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")),
    # International number written with a leading +
    ("PHONE", re.compile(r"\+\d{1,3}[ -]?\(?\d{1,4}\)?(?:[ -]?\d{2,4}){2,4}(?!\d)")),
    ("IP_ADDRESS", re.compile(
        r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.]*\d)")),
    # scheme://user:password@host...  (database / service connection strings)
    ("DATABASE_CREDENTIAL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s@]+@[^\s\"'<>]+")),
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----")),
    # Common API key / token prefixes
    ("API_KEY", re.compile(r"\b(?:sk|pk|rk)[-_](?:live|test)?[-_]?[A-Za-z0-9]{12,}\b")),
    ("API_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("ACCESS_TOKEN", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("ACCESS_TOKEN", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")),
]

# Card numbers: 13-19 digits, optionally separated by spaces or dashes
_CARD_CANDIDATE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")


def _passes_luhn(number: str) -> bool:
    """
    The Luhn check is a simple maths test every real card number passes.
    It stops us from labelling random long numbers as credit cards.
    """
    digits = [int(c) for c in number if c.isdigit()]
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pattern_entities(text: str) -> list[dict]:
    """
    Returns every fixed-format PII value found in `text`, as a list of
    {"text": ..., "type": ..., "reason": ...} dictionaries.
    """
    if not isinstance(text, str) or not text:
        return []

    found: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(value: str, entity_type: str) -> None:
        key = (entity_type, value)
        if value and key not in seen:
            seen.add(key)
            found.append({"text": value, "type": entity_type, "reason": "Matched a fixed PII format"})

    for entity_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            add(match.group(0), entity_type)

    for match in _CARD_CANDIDATE.finditer(text):
        if _passes_luhn(match.group(0)):
            add(match.group(0), "CREDIT_CARD")

    return found
