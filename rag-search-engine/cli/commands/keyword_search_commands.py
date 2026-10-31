import sys

from utils.types import Movie
from utils.text_utils import tokenize_text, tokenize_single_term
from lib import InvertedIndex
from utils.constants import BM25_B

index = InvertedIndex()

def load_index():
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error: Could not find the required file at '{e.filename}'. Please regenerate the cache.")
        sys.exit(1)

def build_command():
    index.build()
    index.save()


def search_command(keyword: str) -> list[Movie]:
    load_index()

    result: list[int] = []
    search_tokens = tokenize_text(keyword, index._stopword_set)
    for token in search_tokens:
        matches = index.get_documents(token)
        result.extend(matches)
        if len(result) >= 5:
            break
    return index.get_movies(result[:5])


def term_frequency_command(keyword: str, doc_id: int):
    load_index()
    token = tokenize_single_term(keyword, index._stopword_set)
    return index.get_tf(doc_id, token)


def inverse_document_frequency_command(keyword: str):
    load_index()
    token = tokenize_single_term(keyword, index._stopword_set)
    return index.get_idf(token)


def tf_idf_command(keyword: str, doc_id: int):
    load_index()
    token = tokenize_single_term(keyword, index._stopword_set)
    term_frequency = index.get_tf(doc_id, token)
    inverse_document_frequency = index.get_idf(token)
    return term_frequency * inverse_document_frequency


def bm25_idf_command(keyword: str):
    load_index()
    token = tokenize_single_term(keyword, index._stopword_set)
    return index.get_bm25_idf(token)


def bm25_tf_command(keyword: str, doc_id: int, k1:float, b=BM25_B):
    load_index()
    token = tokenize_single_term(keyword, index._stopword_set)
    return index.get_bm25_tf(doc_id, token, k1, b)

def bm25_search_command(keyword: str, limit: int):
    load_index()
    return index.bm25_search(keyword, limit)

    