from typing import TypedDict

class MovieScore(TypedDict):
    title: str
    description: str
    score: float


class Movie(TypedDict):
    id: int
    title: str
    description: str


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