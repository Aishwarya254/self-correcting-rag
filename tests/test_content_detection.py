"""Tests for detecting structural book content."""

from self_correcting_rag.content_detection import (
    is_navigation_query,
    is_table_of_contents,
)


def test_detects_navigation_query() -> None:
    assert is_navigation_query("Which chapter discusses AI ethics?") is True


def test_does_not_classify_factual_query_as_navigation() -> None:
    assert is_navigation_query("What is artificial intelligence?") is False


def test_detects_table_of_contents_text() -> None:
    text = (
        "Contents "
        "1 Introduction . . . . . . . . 1 "
        "1.1 What Is AI? . . . . . . . 2 "
        "1.2 Foundations of AI . . . . 5"
    )

    assert is_table_of_contents(text) is True


def test_does_not_classify_ordinary_prose_as_contents() -> None:
    text = (
        "The contents of this chapter explain how intelligent "
        "agents interact with their environments."
    )

    assert is_table_of_contents(text) is False
