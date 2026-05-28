from pydantic import BaseModel
from typing import List


class SimilarProductsResponse(BaseModel):
    product_id: str
    similar_products: List[str]
