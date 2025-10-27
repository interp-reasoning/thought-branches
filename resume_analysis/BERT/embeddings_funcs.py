import numpy as np
from typing import List, Tuple
from BERT.BERT_core_gpu import get_bert_embedding, get_bert_embeddings_batch


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(embedding1.T, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def semantic_similarity(
    text1: str, text2: str, model_name: str = "bert-base-uncased"
) -> float:
    """
    Calculate semantic similarity between two texts using BERT embeddings.

    Args:
        text1: First text
        text2: Second text
        model_name: BERT model to use

    Returns:
        Cosine similarity score between -1 and 1
    """
    embedding1 = get_bert_embedding(text1, model_name)
    embedding2 = get_bert_embedding(text2, model_name)
    return cosine_similarity(embedding1, embedding2)


def find_most_similar(
    query: str,
    candidates: List[str],
    model_name: str = "bert-base-uncased",
    top_k: int = None,
) -> List[Tuple[str, float]]:
    """
    Find the most similar texts to a query using BERT embeddings.

    Args:
        query: Query text
        candidates: List of candidate texts
        model_name: BERT model to use
        top_k: Return only top k results (None for all)

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity
    """
    # Get query embedding
    query_embedding = get_bert_embedding(query, model_name)

    # Get candidate embeddings in batch
    candidate_embeddings = get_bert_embeddings_batch(candidates, model_name)

    # Calculate similarities
    similarities = []
    for text, embedding in zip(candidates, candidate_embeddings):
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((text, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    if top_k is not None:
        return similarities[:top_k]
    return similarities
