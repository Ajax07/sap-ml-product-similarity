import re

import numpy as np
import pandas as pd
from loguru import logger


class ProductPreprocessor:
    """
    Preprocesses product data
    for similarity search.
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        logger.info("Starting preprocessing pipeline")

        df = df.copy()

        # Clean columns
        df = self._clean_brand(df)
        df = self._clean_colour(df)
        df = self._clean_sales_price(df)
        df = self._clean_weight(df)
        df = self._clean_category(df)

        # Build text feature
        df = self._build_text_feature(df)

        logger.info("Finished preprocessing")

        return df

    def _clean_brand(self, df: pd.DataFrame) -> pd.DataFrame:

        df["brand"] = df["brand"].fillna("unknown").astype(str).str.lower().str.strip()

        return df

    def _clean_colour(self, df: pd.DataFrame) -> pd.DataFrame:

        df["colour"] = (
            df["colour"].fillna("unknown").astype(str).str.lower().str.strip()
        )

        return df

    def _clean_sales_price(self, df: pd.DataFrame) -> pd.DataFrame:

        median_price = df["sales_price"].median()

        df["sales_price"] = df["sales_price"].fillna(median_price)

        return df

    def _clean_weight(self, df: pd.DataFrame) -> pd.DataFrame:

        def parse_weight(weight):

            if weight is None:
                return np.nan

            weight = str(weight)

            # Corrupted sentinel
            if weight == "999999999":
                return np.nan

            weight = weight.lower()

            match = re.search(r"(\d+\.?\d*)", weight)

            if not match:
                return np.nan

            value = float(match.group(1))

            # Convert kg → grams
            if "kg" in weight:
                return value * 1000

            return value

        df["weight_cleaned"] = df["weight"].apply(parse_weight)

        median_weight = df["weight_cleaned"].median()

        df["weight_cleaned"] = df["weight_cleaned"].fillna(median_weight)

        return df

    def _clean_category(self, df: pd.DataFrame) -> pd.DataFrame:

        def safe_category(value):

            # None / NaN
            if value is None:
                return ""

            # Float NaN
            if isinstance(value, float):
                if pd.isna(value):
                    return ""

            # Dict → stringify values
            if isinstance(value, dict):
                return " ".join(str(v) for v in value.values())

            # List → stringify
            if isinstance(value, list):
                return " ".join(str(v) for v in value)

            # Everything else
            return str(value)

        df["parent___child_category__all"] = df["parent___child_category__all"].apply(
            safe_category
        )

        return df

    def _build_text_feature(self, df: pd.DataFrame) -> pd.DataFrame:

        product_name = df["product_name"].fillna("").astype(str)

        meta_keywords = df["meta_keywords"].fillna("").astype(str)

        categories = df["parent___child_category__all"].fillna("").astype(str)

        df["combined_text"] = product_name + " " + meta_keywords + " " + categories

        return df
