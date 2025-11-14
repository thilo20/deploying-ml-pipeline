from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Instantiate the app.
app = FastAPI()


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


class Data(BaseModel):
    feature_1: float
    feature_2: str


@app.post("/ingest")
async def ingest_data(data: Data):
    if data.feature_1 < 0:
        raise HTTPException(
            status_code=400, detail={"error": "feature_1 must be non-negative"}
        )
    if len(data.feature_2) > 280:
        raise HTTPException(
            status_code=400,
            detail={"error": "feature_2 must be at most 280 characters long"},
        )

    return {"received_data": data.dict()}
