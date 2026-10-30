from pathlib import Path
import re

from utils.file_utils import load_movies
from utils.chunking import chunk_text, semantic_chunk_text

base_dir = Path(__file__).parent.parent.parent.absolute()
movies_path = base_dir / "data/movies.json"
 
from lib import SemanticSearch, ChunkedSemanticSearch

def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    movies_dict = load_movies(movies_path)
    documents = movies_dict["movies"]
    semantic_search = SemanticSearch()
    embeddings = semantic_search.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def search(query: str, limit: int):
    semantic_search = SemanticSearch()
    movies_dict = load_movies(movies_path)
    documents = movies_dict["movies"]
    semantic_search.load_or_create_embeddings(documents)
    result = semantic_search.search(query, limit)
    index = 1
    for movie_score in result:
        print(f"{index}. {movie_score['title']} (score: {movie_score['score']})")
        print(f"\t{movie_score['description']}")
        index += 1

def chunk(text: str, chunk_size: int, overlap: int):
    return chunk_text(text, chunk_size, overlap)

def semantic_chunk(text: str, max_chunk_size: int, overlap: int):
    return semantic_chunk_text(text, max_chunk_size, overlap)

def embed_chunks():
    movies_dict = load_movies(movies_path)
    documents = movies_dict["movies"]
    semantic_search = ChunkedSemanticSearch()
    embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int):
    movies_dict = load_movies(movies_path)
    documents = movies_dict["movies"]
    chunked_semantic_search = ChunkedSemanticSearch()
    chunked_semantic_search.load_or_create_chunk_embeddings(documents)
    results = chunked_semantic_search.search_chunks(query, limit)
    for index, result in enumerate(results, 1):
        print(f"\n{index}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"   {result['document']}...")