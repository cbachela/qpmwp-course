import os
import joblib
import pandas as pd
from ml.naming.model_name import resolve_target_name
from ml.io.model_io import model_path
import logging

logger = logging.getLogger(__name__)


def train_func(model, X, y, train_idx, target_asset=None, force_retrain=False):
    """
    Trains a model on the specified target asset and saves the trained model to disk.

    Parameters:
    ----------
    model : object
        The machine learning model to be trained. It should have a `.fit()` method and a `.predict()` method.
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for training the model.
    y : pd.DataFrame or pd.Series
        The target labels. If a `pd.DataFrame` is supplied, `target_asset` must be provided.
    train_idx : pd.Index
        The index corresponding to the training period.
    target_asset : str or None
        The asset (column) in `y` that the model will predict. If `y` is a
        `pd.Series`, this parameter is optional; when omitted the function will
        use `y.name` or the fallback string `'target'` as the asset name.
    force_retrain : bool
        If True, forces retraining of the model even if a trained model already exists on disk

    Returns:
    -------
    None
        This function trains the model and saves it to disk, but does not return any value.

    """

    resolved_target = resolve_target_name(y, target_asset)

    path = model_path(X, y, train_idx, target_asset)

    # load from disk if exists already
    if os.path.exists(path) and not force_retrain:
        return

    # restrict features and labels
    X_train = X.loc[train_idx]
    if isinstance(y, pd.Series):
        y_train = y.loc[train_idx]
    else:
        y_train = y.loc[train_idx][resolved_target]

    assert len(X_train) == len(y_train), (len(X_train), len(y_train))

    # fit the model
    model.fit(X=X_train, y=y_train)

    # generate predictions in training set (optional - kept for parity)
    y_pred = pd.DataFrame(model.predict(X=X_train), index=X_train.index, columns=["y_pred"])

    joblib.dump(model, path)
    logger.info("Model trained and saved at %s.", path)


def test_func(X, y, train_idx, test_idx, target_asset=None):
    """
    Tests a trained model on the specified target asset and returns the predictions.

    Parameters:
    ----------
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for making predictions.
    y : pd.DataFrame or pd.Series
        The target labels. If a `pd.DataFrame` is supplied, `target_asset` must be provided.
    train_idx : pd.Index
        The index corresponding to the training period used to identify the trained model.
    test_idx : pd.Index
        The index corresponding to the test period used for predictions.
    target_asset : str or None
        The asset (column) in `y` that the model will predict. If `y` is a
        `pd.Series`, this parameter is optional; when omitted the function will
        use `y.name` or the fallback string `'target'` as the asset name.
    X : pd.DataFrame
        A DataFrame containing the features (independent variables) for making predictions.
    y : pd.DataFrame
        A DataFrame containing the target labels (dependent variables). The target asset should be a column in `y`.
    train_idx : pd.Index
        The index corresponding to the training period. It defines which rows of `X` and `y` were used to train the model.
    test_idx : pd.Index
        The index corresponding to the test period. It defines which rows of `X` and `y` are used for testing the model.

    Returns:
    -------
    tuple
        Always returns a 3-tuple:
            - `resolved_target` : str, the name of the target asset.
            - `y_pred_test` : pd.DataFrame, predicted values for `test_idx`.
            - `y_pred_train` : pd.DataFrame, in-sample predicted values for `train_idx`.
    """
    resolved_target = resolve_target_name(y, target_asset)

    path = model_path(X, y, train_idx, target_asset)

    # load the model from disk, will fail if you don't train one
    model = joblib.load(path)

    # extract the testing part (forecasting)
    # note in live there is no label and we need another function
    X_test = X.loc[test_idx]

    if isinstance(y, pd.Series):
        y_test = y.loc[test_idx]
    else:
        y_test = y.loc[test_idx][resolved_target]

    # prediction dataframe for test set
    y_pred_test = pd.DataFrame(model.predict(X=X_test), index=X_test.index, columns=["y_pred"])

    # also produce in-sample predictions on the training set
    X_train = X.loc[train_idx]
    y_pred_train = pd.DataFrame(model.predict(X=X_train), index=X_train.index, columns=["y_pred"])

    logger.info("Model tested on test set %s to %s. Predictions generated for test and training sets.", test_idx[0], test_idx[-1])

    return (resolved_target, y_pred_test, y_pred_train)
