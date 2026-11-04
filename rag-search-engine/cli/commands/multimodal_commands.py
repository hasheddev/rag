from pathlib import Path

from utils.file_utils import load_movies, get_index_paths
from lib.multimodal_search import MultimodalSearch
from utils.types import MovieDatabase


def get_movies():
    paths = get_index_paths()
    movies_path = paths["movies_path"]
    movies_data: MovieDatabase = load_movies(movies_path)
    movies = movies_data.get("movies", [])
    return movies


def verify_embeddings_command(image: Path):
    if not image.exists():
        raise RuntimeError(f"Image not found for {image}")
    search = MultimodalSearch(get_movies())
    embedding = search.embed_image(image)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image: Path):
    if not image.exists():
        raise RuntimeError(f"Image not found for {image}")
    search = MultimodalSearch(get_movies())
    return search.search_with_image(image)
