"""Segment labels from the source are untrusted. Non-conforming ones are replaced by a stable hash (ticket 038)."""

import hashlib
import re

# fullmatch, not $: a plain $ would let a label with a trailing newline pass unchanged.
_CONFORMING = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _./&-]{0,39}")
# Hex digit to letter: 0 -> a ... 15 -> p. No digits, so the report checks never read a hashed label as a numeral.
_HEX_TO_LETTER = str.maketrans("0123456789abcdef", "abcdefghijklmnop")


def label(value: str) -> str:
    if _CONFORMING.fullmatch(value):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return "label-" + digest.translate(_HEX_TO_LETTER)


def labels(d: dict[str, str]) -> dict[str, str]:
    """Sanitize the values of a segment dict. Keys are trusted metrics-file column names."""
    return {key: label(value) for key, value in d.items()}
