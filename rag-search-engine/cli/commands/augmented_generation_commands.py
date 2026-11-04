from utils.file_utils import get_index_paths, load_movies
from lib.hybrid_search import HybridSearch
from utils.query_utils import (
    generate_response,
    generate_summary,
    generate_response_with_citations,
    generate_direct_response,
)

paths = get_index_paths()


def rag_search(query: str):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    results = hybrid_search.rrf_search(query, 60, None, 5)
    response = generate_response(query, results)
    return response, results


def summarize_rag_search(query: str, limit: int):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    results = hybrid_search.rrf_search(query, 60, None, limit)
    response = generate_summary(query, results)
    return response, results


def rag_search_with_citation(query: str, limit: int):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    results = hybrid_search.rrf_search(query, 60, None, limit)
    response = generate_response_with_citations(query, results)
    return response, results


def answer_question(question: str, limit: int):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    results = hybrid_search.rrf_search(question, 60, None, limit)
    response = generate_direct_response(question, results)
    return response, results
