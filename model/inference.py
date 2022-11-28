import pandas as pd
import joblib
from ml.data import process_data
from ml.model import inference
import json

if __name__ == '__main__':
    df = pd.read_csv("./data/census.csv", skipinitialspace=True)[:2]

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

    model = joblib.load("./outputs/model.pkl")
    encoder = joblib.load("./outputs/encoder.pkl")
    lb = joblib.load("./outputs/lb.pkl")

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        training=False,
        label='salary',
        encoder=encoder,
        lb=lb
    )

    preds = inference(model, X)

    for row in range(df.shape[0]):
        print(json.dumps(df.iloc[row].to_dict()))
        print(preds[row])
