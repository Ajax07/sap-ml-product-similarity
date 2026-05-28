import numpy as np
from loguru import logger

from app.services.data_loader import (
    DataLoader
)

from app.services.preprocessing import (
    ProductPreprocessor
)

from app.services.feature_builder import (
    FeatureBuilder
)

from app.services.text_encoder import (
    TextEncoder
)

from app.services.similarity_engine import (
    SimilarityEngine
)


logger.info(
    "Initializing similarity engine..."
)

loader = DataLoader(
    "data/"
    "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
)

df = loader.load_data()

processor = (
    ProductPreprocessor()
)

processed_df = (
    processor.preprocess(df)
)

builder = (
    FeatureBuilder()
)

structured_features = (
    builder
    .build_structured_features(
        processed_df
    )
)

encoder = TextEncoder()

text_embeddings = (
    encoder.encode(
        processed_df[
            "combined_text"
        ].tolist()
    )
)

structured_weight = 0.45
text_weight = 0.55

final_embeddings = np.hstack([
    structured_features
    * structured_weight,

    text_embeddings
    * text_weight
])

similarity_engine = (
    SimilarityEngine(
        processed_df,
        final_embeddings
    )
)

logger.info(
    "Similarity engine ready."
)