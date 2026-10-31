from pathlib import Path

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from utils.types import (
        FullMovieScore,
        Movie,
        MovieChunkSearchResult,
        HybridWeightedMovieSearch,
        HybridRFFMovieSearch
    )
from utils.hybrid_utils import normalize_scores, hybrid_score, rrf_score


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not self.idx.index_cache_path.exists():
            self.idx.build()
            self.idx.save()
    
    def _semantic_search(self, query: str, limit: int) -> list[MovieChunkSearchResult]:
        return self.semantic_search.search_chunks(query, limit)

    def _bm25_search(self, query: str, limit: int) -> list[FullMovieScore]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridWeightedMovieSearch]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self._semantic_search(query, fetch_limit)

        norm_bm25 = normalize_scores([r["score"] for r in bm25_results])
        norm_semantic = normalize_scores([r["score"] for r in semantic_results])

        result_map: dict[int, HybridWeightedMovieSearch] = {}

        for i, doc in enumerate(bm25_results):
            result_map[doc["id"]] = {
                "id": doc["id"],
                "description": doc["description"],
                "title": doc["title"],
                "hybrid_score": 0.0,
                "keyword_score": norm_bm25[i],
                "semantic_score": 0.0
            }

        for i, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            if doc_id in result_map:
                result_map[doc_id]["semantic_score"] = norm_semantic[i]
            else:
                result_map[doc_id] = {
                    "id": doc_id,
                    "description": doc["document"],
                    "title": doc["title"],
                    "hybrid_score": 0.0,
                    "keyword_score": 0.0,
                    "semantic_score": norm_semantic[i]
                }

        for movie in result_map.values():
            movie["hybrid_score"] = hybrid_score(movie["keyword_score"], movie["semantic_score"], alpha)

        sorted_results = sorted(result_map.values(), key=lambda m: m["hybrid_score"], reverse=True)
        return sorted_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[HybridRFFMovieSearch]:
        fetch_limit = limit * 500
        bm25_results = self._bm25_search(query, fetch_limit)
        semantic_results = self._semantic_search(query, fetch_limit)
        result_map: dict[int, HybridRFFMovieSearch] = {}
        for i, doc in enumerate(bm25_results, 1):
            result_map[doc["id"]] = {
                "id": doc["id"],
                "description": doc["description"],
                "title": doc["title"],
                "keyword_rank": i,
                "rrf_score": 0.0,
                "semantic_rank": 0
            }
        for i, doc in enumerate(semantic_results, 1):
            doc_id = doc["id"]
            if doc_id in result_map:
                result_map[doc_id]["semantic_rank"] = i
            else:
                result_map[doc_id] = {
                    "id": doc_id,
                    "description": doc["document"],
                    "title": doc["title"],
                    "rrf_score": 0.0,
                    "keyword_rank": 0,
                    "semantic_rank": i
                }
        for movie in result_map.values():
            movie["rrf_score"] = rrf_score(movie["keyword_rank"], k) + rrf_score(movie["semantic_rank"], k)
        sorted_results = sorted(result_map.values(), key=lambda m: m["rrf_score"], reverse=True)
        return sorted_results[:limit]
        