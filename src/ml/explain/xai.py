import joblib
import shap
from pathlib import Path
from typing import Optional
from ml.io.model_io import model_path, shap_values_path
import logging

logger = logging.getLogger(__name__)


def _normalize_explainer_type(explainer_type: Optional[str]) -> str:
    if explainer_type is None:
        return "generic"

    normalized = explainer_type.strip().lower()
    aliases = {
        "default": "generic",
        "explainer": "generic",
        "treeexplainer": "tree",
        "linearexplainer": "linear",
        "kernelexplainer": "kernel",
        "permutationexplainer": "permutation",
    }
    normalized = aliases.get(normalized, normalized)

    supported = {"generic", "auto", "tree", "linear", "kernel", "permutation"}
    if normalized not in supported:
        raise ValueError(
            "Unsupported explainer_type "
            f"'{explainer_type}'. Supported values are: "
            "generic, auto, tree, linear, kernel, permutation."
        )

    return normalized


def _build_explainer(actual_model, X_shap, explainer_type: str):
    try:
        if explainer_type == "generic":
            return shap.Explainer(actual_model.predict, X_shap)

        if explainer_type == "auto":
            return shap.Explainer(actual_model, X_shap)

        if explainer_type == "tree":
            return shap.TreeExplainer(actual_model)

        if explainer_type == "linear":
            return shap.LinearExplainer(actual_model, X_shap)

        if explainer_type == "kernel":
            return shap.KernelExplainer(actual_model.predict, X_shap)

        return shap.PermutationExplainer(actual_model.predict, X_shap)
    except Exception as exc:
        raise ValueError(
            f"Could not build a '{explainer_type}' SHAP explainer for "
            f"model type {type(actual_model).__name__}."
        ) from exc


def get_shaply_values(
    X,
    y,
    train_idx,
    target_asset=None,
    explainer_type: Optional[str] = None,
):
    """
    Compute SHAP values for a trained model loaded from disk.

    Args:
        X: Feature data to explain.
        y: Target data for path resolution.
        train_idx: Training index used for path resolution.
        target_asset: Optional target asset name.
        explainer_type: Optional SHAP explainer selector. Supported values are
            generic, auto, tree, linear, kernel, permutation.

    Returns:
        SHAP values for X computed by the explainer.
    """

    explainer_name = _normalize_explainer_type(explainer_type)

    path = model_path(X, y, train_idx, target_asset)

    # load the model from disk, will fail if you don't train one
    model = joblib.load(path)

    if not hasattr(model, "best_estimator_"):
        raise ValueError(
            "Loaded model does not have 'best_estimator_' attribute. "
            "Ensure the model was trained and saved correctly at "
            f"{path}."
        )

    shap_value_path = shap_values_path(
        X,
        y,
        train_idx,
        target_asset,
        explainer_name=explainer_name,
    )

    if Path(shap_value_path).exists():
        print(
            "SHAP values already computed and saved at "
            f"{shap_value_path}. Loading from disk..."
        )
        shap_values = joblib.load(shap_value_path)
        return shap_values

    # Extract pipeline components
    # (loaded model has them)
    pipe = model.best_estimator_

    # get the preprocessing pipeline
    # we need the final features not the raw ones
    preprocessing_pipe = pipe[:-1]

    # get the actual model part
    # for example xgb model itself
    actual_model = pipe[-1]

    if len(pipe.steps) == 1:
        # no preprocessing, using raw data
        X_shap = X.loc[train_idx]
    else:
        # transform the features accordingly
        preprocessing_pipe = pipe[:-1]
        X_shap = preprocessing_pipe.transform(X.loc[train_idx])

    explainer = _build_explainer(actual_model, X_shap, explainer_name)

    # can take a substantial amount of time due to the
    # many perturbations
    shap_values = explainer(X_shap)

    # save the computed SHAP values to disk for future use
    joblib.dump(shap_values, shap_value_path)
    logger.info("SHAP values computed and saved at %s.", shap_value_path)
    
    return shap_values
