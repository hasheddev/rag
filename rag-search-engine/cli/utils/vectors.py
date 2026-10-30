import numpy as np

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def euclidean_norm(vec: list[float]) -> float:
    total = 0.0
    for x in vec:
        total += x**2

    return total**0.5

def add_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    len_a = len(vec1)
    len_b = len(vec2)
    if len_a != len_b:
        raise ValueError(f'Cannot add vectors of different lengths (a={len_a}, b={len_b})')
    result = [0.0] * len_a
    for i in range(len_a):
        result[i] = vec1[i] + vec2[i]
    return result

def subtract_vectors(vec1: list[float], vec2: list[float]) -> list[float]:
    len_a = len(vec1)
    len_b = len(vec2)
    if len_a != len_b:
        raise ValueError(f'cannot subtract vectors of different lengths differ (a={len_a}, b={len_b})')
    result = [0.0] * len_a
    for i in range(len_a):
        result[i] = vec1[i] - vec2[i]
    return result

#The dot product measures how much two vectors point in the same direction. It's calculated by multiplying corresponding elements and summing the results.
def dot(vec1: list[float], vec2: list[float]) -> float:
    len_a = len(vec1)
    len_b = len(vec2)
    if len_a != len_b:
        raise ValueError(f'cannot multiply vectors of different lengths differ (a={len_a}, b={len_b})')
    result = 0.0
    for i in range(len_a):
        result += vec1[i] * vec2[i]
    return result

#Dot product has a problem: it's affected by vector magnitude. Vectors have both direction and magnitude (size). Direction is the important part for semantic similarity, but length can affect vector operations.
#cosine_similarity = dot_product(A, B) / (magnitude(A) * magnitude(B))

def cosine_similarity_mn(vec1: list[float], vec2: list[float]) -> float:
    len_a = len(vec1)
    len_b = len(vec2)
    if len_a != len_b:
        raise ValueError(f'cannot find the cosine similarity of vectors of different lengths differ (a={len_a}, b={len_b})')
    magnitude_a = euclidean_norm(vec1)
    magnitude_b = euclidean_norm(vec2)
    if magnitude_b == 0.0 or magnitude_a == 0.0:
        return 0.0
    return dot(vec1, vec2) / (magnitude_a * magnitude_b)
