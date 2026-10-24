from pathlib import Path
import json
from typing import Callable

def load_movies(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_stopwords(path: Path, func: Callable) -> list[str]:
    return  list(map(func, path.read_text(encoding="utf-8").splitlines()))