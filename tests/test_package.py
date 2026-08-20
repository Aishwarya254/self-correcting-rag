"""Tests for the top-level package."""


def test_package_can_be_imported() -> None:
    import self_correcting_rag

    assert self_correcting_rag.__name__ == "self_correcting_rag"
