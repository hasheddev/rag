from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json

from utils.types import Movie, MovieScore, MovieMetaData, MovieChunkScore, MovieChunkSearchResult
from utils.vectors import cosine_similarity
from utils.chunking import semantic_chunk_text

class SemanticSearch:
    embeddings_cache_file = "movie_embeddings.npy"
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def _store_documents(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {movie["id"]: movie for movie in documents}

    def build_embeddings(self, documents: list[Movie]):
        self._store_documents(documents)
        movie_texts: list[str] = []
        for movie in self.documents:
            movie_texts.append(f"{movie['title']}: {movie['description']}")
        self.embeddings = self.model.encode(movie_texts, show_progress_bar=True)
        cache_dir = Path(__file__).parent.parent.parent / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / self.embeddings_cache_file
        np.save(cache_path, self.embeddings)
        return self.embeddings

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError(f"input text is empty or contains only white space")
        return self.model.encode([text])[0]

    def load_or_create_embeddings(self, documents: list[Movie]):
        self._store_documents(documents)
        cache_dir = Path(__file__).parent.parent.parent / 'cache'
        cache_path = cache_dir / self.embeddings_cache_file
        if cache_path.is_file():
            self.embeddings = np.load(cache_path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        document_score: list[tuple[float, Movie]] = []
        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query_embedding, self.embeddings[i])
            document_score.append((similarity, self.documents[i]))
        sorted_movies = sorted(document_score, key=lambda val: val[0], reverse=True)[:limit]

        result: list[MovieScore] = [
            {
                "score": movie_score[0],
                "title": movie_score[1]["title"],
                "description": movie_score[1]["description"],
            } 
            for movie_score in sorted_movies
        ]
        return result


class ChunkedSemanticSearch(SemanticSearch):
    chunk_embeddings_cache_file ="chunk_embeddings.npy"
    chunk_metadata_file = "chunk_metadata.json"
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[MovieMetaData] = []

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self._store_documents(documents)
        chunks: list[str] = []
        metadata_list: list[MovieMetaData] = []
        doc_length = len(self.documents)
        for index in range(0, doc_length):
           doc = self.documents[index]
           description = doc["description"].strip() 
           if description == "":
               continue
           chunks_list = semantic_chunk_text(description, 4, 1)
           for chunk_idx, chunk in enumerate(chunks_list):
               chunks.append(chunk)
               metadata_list.append({
                   "chunk_idx": chunk_idx,
                   "movie_idx": index,
                   "total_chunks": len(chunks_list)
               })
    
        self.chunk_embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.chunk_metadata = metadata_list
        cache_dir = Path(__file__).parent.parent.parent / 'cache'
        cache_path = cache_dir / self.chunk_embeddings_cache_file
        metadata_path = cache_dir / self.chunk_metadata_file
        np.save(cache_path, self.chunk_embeddings)
        with metadata_path.open("w") as meta_file:
            object = {"chunks": self.chunk_metadata, "total_chunks": len(chunks)}
            json.dump(object, meta_file, indent=2)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]):
        self._store_documents(documents)
        cache_dir = Path(__file__).parent.parent.parent / 'cache'
        cache_path = cache_dir / self.chunk_embeddings_cache_file
        metadata_path = cache_dir / self.chunk_metadata_file
        if cache_path.is_file() and metadata_path.is_file():
            self.chunk_embeddings = np.load(cache_path)
            with metadata_path.open("r") as meta_file:
                object = json.load(meta_file)
                self.chunk_metadata: list[MovieMetaData] = object["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[MovieChunkSearchResult]:
        if self.chunk_embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        chunk_scores: list[MovieChunkScore] = []
        for i in  range(len(self.chunk_embeddings)):
            similarity = cosine_similarity(query_embedding, self.chunk_embeddings[i])
            metadata = self.chunk_metadata[i]
            chunk_scores.append(
                {
                    "chunk_idx": metadata["chunk_idx"],
                    "movie_idx": metadata["movie_idx"],
                    "score": similarity
                }
            )
        movie_best_score: dict[int, MovieChunkScore] = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if movie_idx not in movie_best_score or chunk_score["score"] > movie_best_score[movie_idx]["score"]:
                movie_best_score[movie_idx] = chunk_score
        movie_score_list = list(movie_best_score.values())
        best_matches = sorted(
            movie_score_list, key=lambda meta_data: meta_data["score"], reverse=True
        )[:limit]

        result: list[MovieChunkSearchResult] = []
        for match in best_matches:
            movie = self.documents[match['movie_idx']]
            result.append(
               {
                    "id": movie["id"],
                    "title": movie["title"],
                    "document": movie["description"][:100],
                    "score": round(match["score"], 3),
                }
            )
        return result

