# Script to train machine learning model.
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.model import compute_slice_metrics
import joblib

# Load data.
data = pd.read_csv("./data/census.csv", skipinitialspace=True)

# Train test split
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
    lb=lb,
)

# Train and save a model.
model = train_model(X_train, y_train)

# Inference train and test
preds_train = inference(model, X_train)

# Compute metrics
precision, recall, fbeta = compute_model_metrics(y_train, preds_train)
print(
    f"Train metrics: Precision: {precision:.1%}\
        Recall: {recall:.1%}\
        FBeta: {fbeta:.1%}"
)

preds_test = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds_test)
print(
    f"Test metrics:  Precision: {precision:.1%}\
        Recall: {recall:.1%}\
        FBeta: {fbeta:.1%}"
)

# Compute metrics per slice and saves it
path = 'outputs/slice_output.txt'
if os.path.exists(path):
    os.remove(path)

for col in cat_features:
    df_metrics = compute_slice_metrics(y_test, preds_test, test[col])

    if os.path.exists(path):
        df_metrics.to_csv(path, index=False, mode='a', header=False)
    else:
        df_metrics.to_csv(path, index=False, mode='a', header=True)

# Save model and encoders
joblib.dump(model, 'outputs/model.pkl')
joblib.dump(encoder, 'outputs/encoder.pkl')
joblib.dump(lb, 'outputs/lb.pkl')
