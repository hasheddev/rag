from pathlib import Path
import json
from typing import Callable

from utils.types import MovieDatabase, IndexPaths

def load_movies(path: Path) -> MovieDatabase:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stopwords(path: Path, func: Callable) -> list[str]:
    return  list(map(func, path.read_text(encoding="utf-8").splitlines()))

def get_index_paths() -> IndexPaths:
    base_dir = Path(__file__).parent.parent.parent.absolute()
    movies_path = base_dir / "data/movies.json"
    stopword_path = base_dir / "data/stopwords.txt"
    cache_dir_path = base_dir / "cache"
    return {
        "movies_path": movies_path,
        "cache_dir_path": cache_dir_path,
        "stopword_path": stopword_path
    }
