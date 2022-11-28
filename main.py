from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from model.ml.data import process_data
from model.ml.model import inference

app = FastAPI()


class Data(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example="State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13)
    marital_status: str = Field(example="Never-married")
    occupation: str = Field(example="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example="United-States")


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
