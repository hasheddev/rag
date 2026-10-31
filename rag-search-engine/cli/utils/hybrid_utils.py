def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    list_length = len(scores)
    max_score = max(scores)
    min_score = min(scores)
    if max_score == min_score:
        return [1.0] * list_length
    result = [ (score - min_score) / (max_score - min_score) for score in scores]
    return result

def rrf_score(rank: int, k: int = 60) -> float:
    if rank <= 0:
        return 0.0
    return 1 / (k + rank)