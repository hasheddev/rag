import argparse
from dataclasses import dataclass
from utils.constants import BM25_K1, BM25_B

@dataclass(frozen=True)
class Inputs:
    SEARCH: str = "search"
    BUILD: str = "build"
    TERM_FREQUENCY: str = "tf"
    INVERSE_DOCUMENT_FREQUENCY : str = "idf"
    TFIDF: str = "tfidf"
    BM25_IDF: str = "bm25idf"
    BM25_TF: str = "bm25tf"
    BM25_SEARCH: str = "bm25search"


keyword_search_cli_input = Inputs()

def  init_keyword_search_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    search_parser = subparsers.add_parser(keyword_search_cli_input.SEARCH, help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(keyword_search_cli_input.BUILD, help="Builds the inverted index and caches movie data")

    tf_parser = subparsers.add_parser(keyword_search_cli_input.TERM_FREQUENCY, help="Get the term frequency of a term in a specific document")
    tf_parser.add_argument("doc_id", type=int, help="The ID of the document")
    tf_parser.add_argument("term", type=str, help="The term to check the frequency for")

    idf_parser = subparsers.add_parser(keyword_search_cli_input.INVERSE_DOCUMENT_FREQUENCY, help="Gets the inverse document frequency for a term")
    idf_parser.add_argument("term", type=str, help="The term to check inverse freqency for")

    tfidf_parser = subparsers.add_parser(keyword_search_cli_input.TFIDF, help="Gets the inverse document frequency for a term in a document")
    tfidf_parser.add_argument("doc_id", type=int, help="The ID of the document to use")
    tfidf_parser.add_argument("term", type=str, help="The term to check tfidf for")

    bm25_idf_parser = subparsers.add_parser(
        keyword_search_cli_input.BM25_IDF, help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        keyword_search_cli_input.BM25_TF, help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )

    bm25search_parser = subparsers.add_parser(
        keyword_search_cli_input.BM25_SEARCH, help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")

    bm25search_parser.add_argument(
    "--limit", type=int, nargs='?', default=5, help="Maximum number of results to return (default: 5)"
    )

    return parser