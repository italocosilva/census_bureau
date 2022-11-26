import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier(min_samples_split=350, random_state=23)
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and
    F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds


def compute_slice_metrics(y, preds, slice_column):
    """
    Calculate metrics for each value in slice_column using precision, recall,
    and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    slice_column : pd.Series
        Column used to compute metrics for each unique value.
    Returns
    -------
    metrics : pd.DataFrame
        DataFrame containing precision, recall and fbeta for each value.
    """
    metrics = {
        'column_value': [],
        'precision': [],
        'recall': [],
        'fbeta': []
    }

    for value in slice_column.unique():
        precision, recall, fbeta = compute_model_metrics(
            y[slice_column == value],
            preds[slice_column == value]
        )

        metrics['column_value'].append(slice_column.name + '_' + value)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['fbeta'].append(fbeta)

    return pd.DataFrame(metrics)
