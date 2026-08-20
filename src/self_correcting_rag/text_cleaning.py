"""Clean text extracted from PDF pages."""

import re
import unicodedata

_PARAGRAPH_BREAKS = re.compile(r"\n\s*\n+")
_INLINE_WHITESPACE = re.compile(r"[^\S\n]+")


def clean_extracted_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""

    normalized_text = unicodedata.normalize("NFKC", text)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if not normalized_text:
        return ""

    paragraphs = _PARAGRAPH_BREAKS.split(normalized_text)
    cleaned_paragraphs = []

    for paragraph in paragraphs:
        lines = paragraph.splitlines()
        joined_lines = " ".join(line.strip() for line in lines)
        cleaned_paragraph = _INLINE_WHITESPACE.sub(" ", joined_lines).strip()

        if cleaned_paragraph:
            cleaned_paragraphs.append(cleaned_paragraph)

    return "\n\n".join(cleaned_paragraphs)
