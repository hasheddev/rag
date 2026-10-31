import argparse
from dataclasses import dataclass
from utils.constants import BM25_K1, BM25_B

@dataclass(frozen=True)
class KeywordSearchCommand:
    SEARCH: str = "search"
    BUILD: str = "build"
    TERM_FREQUENCY: str = "tf"
    INVERSE_DOCUMENT_FREQUENCY : str = "idf"
    TFIDF: str = "tfidf"
    BM25_IDF: str = "bm25idf"
    BM25_TF: str = "bm25tf"
    BM25_SEARCH: str = "bm25search"


@dataclass(frozen=True)
class SemanticSearchCommand:
    VERIFY: str = "verify"
    EMBED_TEXT: str = "embed-text"
    VERIFY_EMBEDDINGS: str = "verify-embeddings"
    EMBED_QUERY: str = "embed-query"
    SEARCH: str = "search"
    CHUNK: str = "chunk"
    SEMANTIC_CHUNK: str = "semantic-chunk"
    EMBED_CHUNKS: str = "embed-chunks"
    SEARCH_CHUNKED: str = "search-chunked"

@dataclass(frozen=True)
class HybridSearchCommand:
    NORMALIZE: str = "normalize"
    WEIGHTED_SEARCH:str = "weighted-search"
    RRF_SEARCH:str = "rrf-search"


def  init_keyword_search_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    search_parser = subparsers.add_parser(KeywordSearchCommand.SEARCH, help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser(KeywordSearchCommand.BUILD, help="Builds the inverted index and caches movie data")

    tf_parser = subparsers.add_parser(
        KeywordSearchCommand.TERM_FREQUENCY, help="Get the term frequency of a term in a specific document"
    )
    tf_parser.add_argument("doc_id", type=int, help="The ID of the document")
    tf_parser.add_argument("term", type=str, help="The term to check the frequency for")

    idf_parser = subparsers.add_parser(
        KeywordSearchCommand.INVERSE_DOCUMENT_FREQUENCY, help="Gets the inverse document frequency for a term"
    )
    idf_parser.add_argument("term", type=str, help="The term to check inverse freqency for")

    tfidf_parser = subparsers.add_parser(
        KeywordSearchCommand.TFIDF, help="Gets the inverse document frequency for a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="The ID of the document to use")
    tfidf_parser.add_argument("term", type=str, help="The term to check tfidf for")

    bm25_idf_parser = subparsers.add_parser(
        KeywordSearchCommand.BM25_IDF, help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
        KeywordSearchCommand.BM25_TF, help="Get BM25 TF score for a given document ID and term"
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
        KeywordSearchCommand.BM25_SEARCH, help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit",
        type=int, 
        nargs='?',
        default=5, help="Maximum number of results to return (default: 5)"
    )

    return parser


def init_semantic_search_parser():
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser(SemanticSearchCommand.VERIFY, help="Prints information about the semantic model used")

    embed_parser = subparsers.add_parser(
        SemanticSearchCommand.EMBED_TEXT, help="Embeds a given text and prints part of the vector embedding"
    )
    embed_parser.add_argument("term", help="Text to embed")

    subparsers.add_parser(SemanticSearchCommand.VERIFY_EMBEDDINGS, help="Loads or embeds and caches movies")

    embed_query_parser = subparsers.add_parser(
        SemanticSearchCommand.EMBED_QUERY, help="Embeds a given text and prints part of the vector embedding"
    )
    embed_query_parser.add_argument("term", help="Text to embed")

    search_parser = subparsers.add_parser(
        SemanticSearchCommand.SEARCH, help="Search movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, nargs='?', default=5, help="Maximum number of results to return (default: 5)"
    )

    chunk_parser = subparsers.add_parser(
        SemanticSearchCommand.CHUNK,
        help="splits given text into smaller chunks of given size(default 200)"
    )
    chunk_parser.add_argument("text", type=str, help="The text string to chunk")
    chunk_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=200, 
        help="Number of words per chunk (default: 200)"
    )
    chunk_parser.add_argument(
        "--overlap", 
        type=int, 
        default=0, 
        help="Length of chunk overlapping with previous chunk (default: 0)"
    )

    semantic_chunk_parser = chunk_parser = subparsers.add_parser(
        SemanticSearchCommand.SEMANTIC_CHUNK, help="splits given text into smaller chunks of given size(default 200)"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="The text string to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", 
        type=int, 
        default=4, 
        help="Maximum number of sentences per chunk (default: 4)"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", 
        type=int, 
        default=0, 
        help="Number of sentences in a chunk overlapping with previous chunk (default: 0)"
    )

    subparsers.add_parser(
        SemanticSearchCommand.EMBED_CHUNKS,
        help="Embeds movies by chunking movie description into chunks of four sentences with overlap of one sentence"
    )

    search_chunked_parser = subparsers.add_parser(
        SemanticSearchCommand.SEARCH_CHUNKED,
        help="Uses chunk cache to search for documents that best match query"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, nargs='?', default=5, help="Maximum number of results to return (default: 5)"
    )

    return parser

def init_hybrid_search_parser():
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        HybridSearchCommand.NORMALIZE, 
        help="Normalize a list of scores"
    )

    normalize_parser.add_argument(
        "scores", 
        type=float, 
        nargs="*", 
        help="List of scores to normalize (e.g., 0.5 2.3 1.2)"
    )

    weighted_parser = subparsers.add_parser(
        HybridSearchCommand.WEIGHTED_SEARCH,
        help="Uses weighted search technique to search for documents that best match query"
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument(
        "--limit",
        type=int, 
        nargs='?',
        default=5, help="Maximum number of results to return (default: 5)"
    )
    weighted_parser.add_argument(
        "--alpha",
        type=float, 
        nargs='?',
        default=0.5, help="Alpha ratio of semantic to keyword search for scores weights (default: 0.5)(0.2==keyword search 20%)"
    )

    rrf_parser = subparsers.add_parser(
        HybridSearchCommand.RRF_SEARCH,
        help="Uses Reciprocal Rank Fusion to search for documents that best match query"
    )
    rrf_parser.add_argument("query", type=str, help="Search query")
    rrf_parser.add_argument(
        "--limit",
        type=int, 
        default=5,
        help="Maximum number of results to return (default: 5)"
    )
    rrf_parser.add_argument(
        "--k",
        type=int, 
        default=60,
        help="Value used to calculate Reciprocal Rank Fusion score (default: 60)"
    )
    
    return parser