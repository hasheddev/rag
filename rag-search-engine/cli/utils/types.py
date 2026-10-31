from typing import TypedDict
from pathlib import Path

class MovieScore(TypedDict):
    title: str
    description: str
    score: float


class Movie(TypedDict):
    id: int
    title: str
    description: str


class FullMovieScore(TypedDict):
    id: int
    title: str
    description: str
    score: float


class MovieDatabase(TypedDict):
    movies: list[Movie]


class MovieMetaData(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class MovieChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float

class MovieChunkSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float

class IndexPaths(TypedDict):
    movies_path: Path
    stopword_path: Path
    cache_dir_path: Path


class HybridWeightedMovieSearch(Movie):
    keyword_score: float
    semantic_score: float
    hybrid_score: float

class HybridRFFMovieSearch(Movie):
    keyword_rank: int
    semantic_rank: int
    rrf_score: float