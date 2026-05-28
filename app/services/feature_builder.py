import numpy as np
import pandas as pd

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from scipy.sparse import hstack


class FeatureBuilder:
    """
    Builds structured features.
    """

    def __init__(self):

        self.scaler = (
            StandardScaler()
        )

        self.encoder = (
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    def build_structured_features(
        self,
        df: pd.DataFrame
    ) -> np.ndarray:

        numeric_cols = [
            "sales_price",
            "rating",
            "weight_cleaned"
        ]

        categorical_cols = [
            "brand",
            "colour",
            "delivery_type",
            "amazon_prime__y_or_n",
            "best_seller_tag__y_or_n",
        ]

        numeric_features = (
            self.scaler.fit_transform(
                df[numeric_cols]
            )
        )

        categorical_features = (
            self.encoder.fit_transform(
                df[categorical_cols]
            )
        )

        structured_features = hstack([
            numeric_features,
            categorical_features
        ])

        return (
            structured_features
            .toarray()
        )