import faiss
import numpy as np
from loguru import logger


class FaissIndex:
    """
    Handles FAISS indexing
    and nearest neighbor search.
    """

    def __init__(self):

        self.index = None

    def build_index(self, embeddings: np.ndarray):

        logger.info("Building FAISS index")

        embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        logger.info(f"Indexed " f"{len(embeddings)} " f"products")

    def search(self, query_vector: np.ndarray, k: int = 5):

        query_vector = query_vector.astype("float32").reshape(1, -1)

        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        return (scores[0], indices[0])
