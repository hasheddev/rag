from collections import Counter
from pathlib import Path
import pickle
import math

from utils.text_utils import  tokenize_text, remove_punctuations
from utils.file_utils import load_movies, load_stopwords
from utils.constants import BM25_K1, BM25_B
from utils.types import MovieDatabase, Movie

class InvertedIndex:
    index_path = 'index.pkl'
    docmap_path = 'docmap.pkl'
    doc_lengths_path = 'doc_lengths.pkl'
    term_frequency_path = 'term_frequencies.pkl'

    def __init__(self, cache_path: Path, movies_path: Path, stopwords_path: Path) -> None:
        self._path = cache_path
        self._index: dict[str, set[int]] =  {}
        self._docmap: dict[int, Movie] = {}
        self._term_frequencies: dict[int, Counter] = {}
        self._doc_lengths: dict[int, int] = {}
        self._movies_path = movies_path
        self._stopwords_path = stopwords_path
        raw_stopwords = load_stopwords(self._stopwords_path, remove_punctuations)
        self._stopword_set = set(raw_stopwords)
        

    @property
    def docmap(self):
        return self._docmap

    @property
    def doc_lengths(self):
        return self._doc_lengths
    
    @property
    def index(self):
        return self._index

    @property
    def term_frequencies(self):
        return self._term_frequencies

    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text, self._stopword_set)
        token_count = len(tokens)
        self._doc_lengths[doc_id] = token_count
        for token in tokens:
            self._index.setdefault(token, set()).add(doc_id)
            self._term_frequencies.setdefault(doc_id, Counter())[token] += 1

    def __get_avg_doc_length(self) -> float:
        total_doc_count = len(self._doc_lengths)
        if total_doc_count == 0:
            return 0.0
        return sum(self._doc_lengths.values()) / total_doc_count

    def bm25(self, doc_id: int, term: str):
        bm25_idf = self.get_bm25_idf(term)
        bm25_tf = self.get_bm25_tf(doc_id, term)

        return bm25_idf * bm25_tf

    def bm25_search(self, query: str, limit: int):
        tokens = tokenize_text(query, self._stopword_set)
        scores: dict[int, float] = {}

        for token in tokens:
            matches = self.get_documents(token)
            for doc_id in matches:
                scores.setdefault(doc_id, 0.0)
                scores[doc_id] += self.bm25(doc_id, token)
        values = sorted(scores.items(), key=lambda v: v[1], reverse=True)[:limit]
        movies_with_bm25_score = [(movie, value[1]) for value in values if (movie := self._docmap.get(value[0])) is not None]
        return movies_with_bm25_score
        
 
    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self._docmap)
        term_matches = self._index.get(term, set())
        term_match_doc_count = len(term_matches)

        return math.log(((total_doc_count - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1))

    def get_idf(self, term: str) -> float:
        total_doc_count = len(self._docmap)
        term_matches = self._index.get(term, set())
        term_match_doc_count = len(term_matches)
        
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_documents(self, term: str):
        matches = self._index.get(term, set())
        return sorted(list(matches))

    def get_movies(self, doc_ids: list[int]) -> list[Movie]:
        return  [movie for doc_id in doc_ids if (movie := self._docmap.get(doc_id)) is not None]

    def get_bm25_tf(self, doc_id: int, term: str, k1: float =BM25_K1, b=BM25_B):
        counter = self._term_frequencies.get(doc_id, Counter())
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self._doc_lengths.get(doc_id, 0)
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf =  counter[term]
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def get_tf(self, doc_id: int, term: str):
        counter = self._term_frequencies.get(doc_id, Counter())
        return counter[term]

    def build(self):
        movies_data: MovieDatabase = load_movies(self._movies_path)
        movies = movies_data.get("movies", [])

        if not movies:
            return
        for movie in movies:
            doc_id = movie["id"]
            self._docmap[doc_id] = movie
            
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
    
    def save(self):
        self._path.mkdir(parents=True, exist_ok=True)
        index_cache_path = self._path / self.index_path
        docmap_cache_path = self._path / self.docmap_path
        doclength_cache_path = self._path / self.doc_lengths_path
        term_frequency_cache_path = self._path / self.term_frequency_path

        with index_cache_path.open("wb") as index_file:
            pickle.dump(self._index, index_file)

        with docmap_cache_path.open("wb") as docmap_file:
            pickle.dump(self._docmap, docmap_file)

        with term_frequency_cache_path.open("wb") as term_frequency_file:
            pickle.dump(self._term_frequencies, term_frequency_file)

        with doclength_cache_path.open("wb") as doc_length_file:
            pickle.dump(self._doc_lengths, doc_length_file)
        

    def load(self):
        index_cache_path = self._path / self.index_path
        docmap_cache_path = self._path / self.docmap_path
        doclength_cache_path = self._path / self.doc_lengths_path
        term_frequency_cache_path = self._path / self.term_frequency_path

        with index_cache_path.open("rb") as index_file:
            self._index = pickle.load(index_file)
        
        with docmap_cache_path.open("rb") as docmap_file:
            self._docmap = pickle.load(docmap_file)

        with term_frequency_cache_path.open("rb") as term_frequency_file:
            self._term_frequencies = pickle.load(term_frequency_file)

        with doclength_cache_path.open("rb") as doc_length_file:
            self._doc_lengths = pickle.load(doc_length_file)

