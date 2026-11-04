from lib.hybrid_search import HybridSearch
from utils.types import enhance_type, rerank_type
from utils.file_utils import get_index_paths, load_movies
from utils.hybrid_utils import normalize_scores
from utils.query_utils import (
    enhance_query_spelling,
    rewrite_user_query,
    expand_user_query,
    rerank_individual,
    rerank_batches,
    rerank_with_cross_encoder,
    print_rrf_results,
    evaluate_movie_results,
)

paths = get_index_paths()


def normalize(scores: list[float]) -> list[float]:
    return normalize_scores(scores)


def weighted_search(query: str, limit: int, alpha: float):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    return hybrid_search.weighted_search(query, alpha, limit)


# Reciprocal Rank Fusion
def rrf_search(
    query: str,
    limit: int,
    k: int,
    enhance_opt: enhance_type,
    rerank_opt: rerank_type,
    evaluate_result: bool | None,
):
    enhanced_query: str | None = None
    print(f"Original query: {query}")
    match enhance_opt:
        case "spell":
            enhanced_query = enhance_query_spelling(query)
        case "rewrite":
            enhanced_query = rewrite_user_query(query)
        case "expand":
            enhanced_query = expand_user_query(query)
        case _:
            pass
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    search_str = enhanced_query if enhanced_query is not None else query
    matches = hybrid_search.rrf_search(search_str, k, rerank_opt, limit)
    print(f"Reciprocal Rank Fusion Results for '{search_str}' (k={k}):\n")
    print("Results After RRF Search")
    print("\n*************")
    print_rrf_results(matches, rerank_opt)
    print("*************\n")
    if evaluate_result:
        print("\n*************")
        print("Evaluating Result...\n")
        evaluate_movie_results(query, matches)
        print("*************\n")

    match rerank_opt:
        case "individual":
            new_result = rerank_individual(search_str, matches)
            sorted_result = sorted(
                new_result, reverse=True, key=lambda movie: movie["llm_score"]
            )[:limit]
            print(f"Re-ranking top {limit} results using individual method...")
            return sorted_result
        case "batch":
            new_result = rerank_batches(search_str, matches)
            print(f"Re-ranking top {limit} results using batch method...")
            return new_result[:limit]
        case "cross_encoder":
            print(f"Re-ranking top {limit} results using cross_encoder method...")
            new_results = rerank_with_cross_encoder(search_str, matches)
            return new_results[:limit]
        case _:
            return matches[:limit]
