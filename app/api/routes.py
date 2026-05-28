from fastapi import APIRouter, HTTPException

from app.core.engine import similarity_engine

router = APIRouter()


@router.get("/find_similar_products")
def get_similar_products(product_id: str, num_similar: int = 5):

    try:

        similar_products = similarity_engine.find_similar_products(
            product_id=product_id, num_similar=num_similar
        )

        return {"product_id": product_id, "similar_products": similar_products}

    except ValueError as e:

        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))
