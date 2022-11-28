from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from model.ml.data import process_data
from model.ml.model import inference

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/")
async def get_items():
    return {"Hello, welcome to Census Data Salary Prediction API."}


@app.post("/inference/")
async def predict(data: Data):
    df = pd.Series(data.dict()).to_frame().T

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    model = joblib.load("./outputs/model.pkl")
    encoder = joblib.load("./outputs/encoder.pkl")
    lb = joblib.load("./outputs/lb.pkl")

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    pred = {"pred": int(inference(model, X)[0])}
    return pred
