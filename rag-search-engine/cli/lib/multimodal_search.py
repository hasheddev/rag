from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path

from utils.types import Movie, FullMovieScore
from utils.vectors import cosine_similarity


class MultimodalSearch:
    def __init__(self, docs: list[Movie], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = docs
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in docs]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: Path):
        image = Image.open(image_path)
        return self.model.encode([image])[0]

    def search_with_image(self, image_path: Path):
        image_embedding = self.embed_image(image_path)
        document_score: list[tuple[float, Movie]] = []

        for i in range(len(self.text_embeddings)):
            similarity = cosine_similarity(image_embedding, self.text_embeddings[i])
            document_score.append((similarity, self.documents[i]))

        sorted_movies = sorted(document_score, key=lambda val: val[0], reverse=True)[:5]
        result: list[FullMovieScore] = [
            {
                "id": movie_score[1]["id"],
                "title": movie_score[1]["title"],
                "description": movie_score[1]["description"],
                "score": movie_score[0],
            }
            for movie_score in sorted_movies
        ]
        return result
