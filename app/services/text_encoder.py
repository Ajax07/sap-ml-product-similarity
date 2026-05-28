from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np


class TextEncoder:
    """
    Encodes product text
    using sentence embeddings.
    """

    def __init__(self, model_name: str = ("sentence-transformers/" "all-MiniLM-L6-v2")):

        logger.info(f"Loading model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:

        logger.info(f"Encoding {len(texts)} products")

        embeddings = self.model.encode(
            texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
        )

        return embeddings
