import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class RagCommand:
    RAG: str = "rag"
    SUMMARIZE: str = "summarize"
    CITATIONS: str = "citations"
    QUESTION: str = "question"


@dataclass(frozen=True)
class MultiModalCommand:
    VERIFY_EMBEDDING: str = "verify-embedding"
    IMAGE_SEARCH: str = "image-search"


def init_rag_search_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        RagCommand.SUMMARIZE,
        help="Perform RAG (search + generate answer) and summarize result",
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    citations_parser = subparsers.add_parser(
        RagCommand.CITATIONS,
        help="Perform RAG (search + generate answer) and cites answer sources",
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    question_parser = subparsers.add_parser(
        RagCommand.QUESTION,
        help="Perform RAG (search + generate answer) and responds to user questions",
    )
    question_parser.add_argument("question", type=str, help="Search query")
    question_parser.add_argument(
        "--limit",
        type=int,
        nargs="?",
        default=5,
        help="Maximum number of results to return (default: 5)",
    )

    return parser


def init_multimodal_search_parser():
    parser = argparse.ArgumentParser(
        description="Rewrite a text query based on an image."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_parser = subparsers.add_parser(
        MultiModalCommand.VERIFY_EMBEDDING,
        help="Embeds an image and verifies its success",
    )
    image_parser.add_argument(
        "--image-path", required=True, help="Path to the image file."
    )

    image_search = subparsers.add_parser(
        MultiModalCommand.IMAGE_SEARCH, help="Searches movies based on image"
    )
    image_search.add_argument(
        "--image-path", required=True, help="Path to the image file."
    )

    return parser
