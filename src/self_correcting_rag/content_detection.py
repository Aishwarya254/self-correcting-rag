"""Detect structural and navigational content in books."""

import re

_CONTENTS_HEADING = re.compile(
    r"\b(?:table\s+of\s+)?contents\b",
    re.IGNORECASE,
)
_DOT_LEADER = re.compile(r"(?:\.\s*){4,}")
_NUMBERED_SECTION = re.compile(r"\b\d+(?:\.\d+)+\s+\w+")
_NAVIGATION_QUERY = re.compile(
    r"\b(?:which|what)\s+(?:chapter|section|page)\b"
    r"|\bwhere\s+(?:is|are|can|does|do)\b",
    re.IGNORECASE,
)


def is_navigation_query(text: str) -> bool:
    """Return whether text appears to be a navigation query."""

    return bool(_NAVIGATION_QUERY.search(text))


def is_table_of_contents(text: str) -> bool:
    """Return whether text appears to contain a table of contents."""

    if not _CONTENTS_HEADING.search(text):
        return False

    dot_leaders = len(_DOT_LEADER.findall(text))
    numbered_sections = len(_NUMBERED_SECTION.findall(text))

    return dot_leaders >= 2 or numbered_sections >= 2
