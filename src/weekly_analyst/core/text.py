"""Numerals as a reader sees them. Moved from checks.py so report and analysis share it from core."""

import re

# A numeral as a reader sees it: optional sign, plain or grouped integer, optional
# decimal part, optional K/M/B/% suffix. The lookarounds stop matches inside words
# and codes, so "Q3" and "W02" give no numeral.
_NUMERAL = re.compile(r"(?<![\w.])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?[KMB%]?(?!\w)")


def numerals(text: str) -> list[str]:
    """Every numeral a reader sees in text, in order, repeats included."""
    return _NUMERAL.findall(text)
