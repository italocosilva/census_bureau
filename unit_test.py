import pytest
import json
from model.ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier
from fastapi.testclient import TestClient
from main import app


def test_train():
    X_train = [[0], [0], [1], [1]]
    y_train = [0, 0, 1, 1]

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_metrics():
    y = [0, 0, 1, 1]
    preds = [0, 1, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert (precision == 0.5) & (recall == 0.5) & (fbeta == 0.5)


@pytest.fixture
def model():
    X_train = [[0], [0], [1], [1]]
    y_train = [0, 0, 1, 1]

    model = train_model(X_train, y_train)

    return model


def test_inference(model):
    X = [[0], [0], [1], [1]]
    preds = inference(model, X)

    assert len(preds) == len(X)


client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["Hello, welcome to Census Data Salary Prediction API."]


def test_post_1():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"pred": 0}


def test_post_2():
    data = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 13,
        "native_country": "United-States"
    }

    r = client.post("/inference/", data=json.dumps(data))
    assert r.status_code == 200
    assert r.json() == {"pred": 1}
