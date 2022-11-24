# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import *
from ml.model import *

# Load data.
data = pd.read_csv('./data/census.csv', skipinitialspace=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, 
    categorical_features=cat_features, 
    label="salary", 
    training=False,
    encoder=encoder,
    lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Inference train and test
preds_train = inference(model, X_train)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_train, preds_train)
print(f'Train metrics: Precision: {precision:.1%}    Recall: {recall:.1%}    FBeta: {fbeta:.1%}')

preds_test = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds_test)
print(f'Test metrics:  Precision: {precision:.1%}    Recall: {recall:.1%}    FBeta: {fbeta:.1%}')
