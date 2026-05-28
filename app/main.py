from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="SAP Product Similarity API")

app.include_router(router)


@app.get("/")
def health_check():

    return {"message": "SAP Product " "Similarity API " "is running"}
