"""Tests for cleaning text extracted from PDFs."""

from self_correcting_rag.text_cleaning import clean_extracted_text


def test_clean_extracted_text_normalizes_spacing_and_paragraphs() -> None:
    raw_text = (
        "  Retrieval   augmented\tgeneration\r\n"
        "uses external evidence.  \r\n"
        "\r\n"
        "  A new paragraph starts here.  "
    )

    cleaned_text = clean_extracted_text(raw_text)

    assert cleaned_text == (
        "Retrieval augmented generation uses external evidence.\n\nA new paragraph starts here."
    )


def test_clean_extracted_text_handles_empty_content() -> None:
    assert clean_extracted_text("") == ""
    assert clean_extracted_text(" \n\t\r\n ") == ""
