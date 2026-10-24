from typing import TypedDict

class Movie(TypedDict):
    id: int
    title: str
    description: str


class MovieDatabase(TypedDict):
    movies: list[Movie]