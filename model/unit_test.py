import pytest
from .ml.model import train_model, compute_model_metrics, inference
from sklearn.ensemble import RandomForestClassifier


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
