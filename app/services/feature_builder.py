import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureBuilder:
    """
    Builds lightweight structured
    features for similarity.
    """

    def __init__(self):

        self.scaler = StandardScaler()

        self.brand_encoder = LabelEncoder()

        self.colour_encoder = LabelEncoder()

    def build_structured_features(self, df: pd.DataFrame) -> np.ndarray:

        # Numeric features
        numeric_cols = ["sales_price", "rating", "weight_cleaned"]

        numeric_features = self.scaler.fit_transform(df[numeric_cols])

        # Encode categorical features
        brand_encoded = self.brand_encoder.fit_transform(df["brand"]).reshape(-1, 1)

        colour_encoded = self.colour_encoder.fit_transform(df["colour"]).reshape(-1, 1)

        prime_encoded = (
            (df["amazon_prime__y_or_n"] == "Y").astype(int).values.reshape(-1, 1)
        )

        bestseller_encoded = (
            (df["best_seller_tag__y_or_n"] == "Y").astype(int).values.reshape(-1, 1)
        )

        structured_features = np.hstack(
            [
                numeric_features,
                brand_encoded,
                colour_encoded,
                prime_encoded,
                bestseller_encoded,
            ]
        )

        return structured_features.astype(np.float32)
