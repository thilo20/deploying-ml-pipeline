import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starter.ml.data import process_data
from starter.ml.model import inference

# Instantiate the app.
app = FastAPI()


# static welcome message
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


def load_model_artifacts():
    """Load model, encoder, and label binarizer from disk."""
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("model/lb.pkl", "rb") as f:
        lb = pickle.load(f)
    return model, encoder, lb


# Load model at startup
model, encoder, lb = load_model_artifacts()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    # data example: these values match the 1st row of census.csv
    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
                # "salary": "<=50K" # prediction target, not part of input
            }
        }


@app.post("/predict")
async def predict(data: CensusData):
    # Convert input to DataFrame
    input_dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country],
    }

    df = pd.DataFrame(input_dict)

    # Process data
    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # predict salary (binary)
    pred = inference(model, X)

    # Convert prediction to label
    result = "<=50K"
    if pred[0] == 1:
        result = ">50K"

    return {"prediction": result}
