import numpy as np

def compute_top_k(query_embedding, article_embeddings, k=10) -> np.ndarray:
    # Compute cosine similarity
    cos_sims = (query_embedding * article_embeddings).sum(axis=1) / (np.linalg.norm(query_embedding) * np.linalg.norm(article_embeddings, axis=1))
    top_k_indices = np.argsort(cos_sims)[-k:][::-1]
    return top_k_indices