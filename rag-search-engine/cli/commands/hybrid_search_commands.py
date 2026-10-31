from lib.hybrid_search import HybridSearch
from utils.types import Movie
from utils.file_utils import get_index_paths, load_movies
from utils.hybrid_utils import normalize_scores

paths = get_index_paths()

def normalize(scores: list[float]) -> list[float]:
    return normalize_scores(scores)

def weighted_search(query:str, limit: int, alpha: float):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    return hybrid_search.weighted_search(query, alpha, limit)

#Reciprocal Rank Fusion
def rff_search(query:str, limit: int, k: int):
    movies_dict = load_movies(paths["movies_path"])
    hybrid_search = HybridSearch(movies_dict["movies"])
    return hybrid_search.rrf_search(query, k, limit)