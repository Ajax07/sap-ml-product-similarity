import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import (
    cosine_similarity
)


class SimilarityEngine:
    """
    Product similarity engine
    using cosine similarity.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.embeddings = (
            embeddings
        )

        # Product lookup
        self.product_id_to_idx = {
            product_id: idx
            for idx, product_id
            in enumerate(
                self.df["uniq_id"]
            )
        }

    def find_similar_products(
        self,
        product_id: str,
        num_similar: int = 5
    ) -> list[str]:

        if (
            product_id
            not in self.product_id_to_idx
        ):
            raise ValueError(
                f"Product ID "
                f"{product_id} "
                f"not found."
            )

        query_idx = (
            self.product_id_to_idx[
                product_id
            ]
        )

        query_vector = (
            self.embeddings[
                query_idx
            ]
            .reshape(1, -1)
        )

        similarities = (
            cosine_similarity(
                query_vector,
                self.embeddings
            )[0]
        )

        sorted_indices = (
            np.argsort(
                similarities
            )[::-1]
        )

        query_name = (
            self.df.iloc[
                query_idx
            ]["product_name"]
        )

        seen_products = set()

        similar_products = []

        for idx in sorted_indices:

            candidate_row = (
                self.df.iloc[idx]
            )

            candidate_id = (
                candidate_row[
                    "uniq_id"
                ]
            )

            candidate_name = (
                candidate_row[
                    "product_name"
                ]
            )

            # Skip self
            if (
                candidate_id
                == product_id
            ):
                continue

            # Skip exact duplicate names
            if (
                candidate_name
                == query_name
            ):
                continue

            # Avoid duplicates
            if (
                candidate_name
                in seen_products
            ):
                continue

            seen_products.add(
                candidate_name
            )

            similar_products.append(
                candidate_id
            )

            if (
                len(similar_products)
                >= num_similar
            ):
                break

        return similar_products