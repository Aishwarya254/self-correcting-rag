"""Command-line interface for indexing and searching PDF books."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from self_correcting_rag.embeddings import SentenceTransformerEmbedder
from self_correcting_rag.pipeline import index_pdf


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="self-rag",
        description="Search PDF books using semantic retrieval.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search",
        help="Index a PDF and retrieve evidence for a question.",
    )
    search_parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the PDF book.",
    )
    search_parser.add_argument(
        "query",
        help="Question used to retrieve relevant evidence.",
    )
    search_parser.add_argument("--limit", type=int, default=4)
    search_parser.add_argument("--chunk-size", type=int, default=300)
    search_parser.add_argument("--overlap", type=int, default=50)
    search_parser.add_argument("--device", default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "search":
        embedder = SentenceTransformerEmbedder(device=args.device)
        indexed_book = index_pdf(
            args.pdf,
            embedder,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
        results = indexed_book.search(
            args.query,
            limit=args.limit,
        )

        print("# Retrieved Evidence")

        if not results:
            print("\nNo relevant evidence was found.")
            return 0

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            pages = (
                str(chunk.start_page)
                if chunk.start_page == chunk.end_page
                else f"{chunk.start_page}-{chunk.end_page}"
            )

            print(f"\n## Result {rank}")
            print(f"- **Source:** {chunk.source}")
            print(f"- **Pages:** {pages}")
            print(f"- **Score:** {result.score:.4f}")
            print(f"- **Text:** {chunk.text}")

        return 0

    parser.error(f"unsupported command: {args.command}")
