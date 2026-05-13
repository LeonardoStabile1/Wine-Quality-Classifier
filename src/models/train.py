from pathlib import Path
import json
from typing import Dict, Any, Tuple

import joblib
from loguru import logger
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import pandas as pd


def train_xgboost(
    X_train,
    y_train,
    X_val,
    y_val,
    n_iter: int = 40,
    cv: int = 3,
    random_state: int = 42,
    skip_random_search: bool = False,
) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Train an XGBoost classifier using randomized hyperparameter search.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        n_iter: Number of parameter settings sampled.
        cv: Number of cross-validation folds.
        random_state: Random seed.
        skip_random_search:
            If True, skip hyperparameter search.

    Returns:
        Tuple containing trained model and best parameters.
    """

    if X_train is None or y_train is None:
        raise ValueError("X_train and y_train must not be None")

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training data cannot be empty")

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    logger.info(
        "Starting XGBoost training | samples={} features={}",
        len(X_train),
        X_train.shape[1],
    )

    # =========================================================
    # SIMPLE TRAINING
    # =========================================================

    if skip_random_search:
        best_params = {
            "n_estimators": 800,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        }

        model = XGBClassifier(
            **best_params,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info("Simple XGBoost training finished")

        return model, best_params

    # =========================================================
    # RANDOM SEARCH
    # =========================================================

    param_dist = {
        "n_estimators": randint(200, 800),
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": randint(3, 8),
        "min_child_weight": randint(1, 8),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "gamma": uniform(0.0, 0.3),
        "reg_alpha": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.5, 1.5),
    }

    base_model = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    logger.info("Starting Random Search")

    search.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    best_params = search.best_params_

    logger.info("Random Search finished")
    logger.info("Best params: {}", best_params)

    final_model = XGBClassifier(
        **best_params,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )

    final_model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    logger.info("Final model training finished")

    return final_model, best_params


def save_artifacts(
    path: str,
    X_train: pd.DataFrame,
    model_name: str,
    model,
    params: Dict[str, Any],
    scaler=None,
) -> None:
    """
    Persist model artifacts to disk.

    Args:
        path: Directory path where artifacts will be saved.
        model_name: File name for the saved artifact.
        model: Trained model instance.
        params: Model hyperparameters.
        scaler: Optional scaler used in preprocessing.

    Returns:
        None
    """
    if not path:
        logger.error("Invalid path provided: {}", path)
        raise ValueError("Path must be a non-empty string")

    if not model_name or not isinstance(model_name, str):
        logger.error("Invalid model_name: {}", model_name)
        raise ValueError("model_name must be a valid string")

    if model is None:
        logger.error("Model cannot be None")
        raise ValueError("Model must not be None")

    if not isinstance(params, dict):
        logger.error("Params must be a dictionary")
        raise ValueError("params must be a dictionary")

    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)

    file_path = base_path / model_name

    artifacts = {
        "model": model,
        "scaler": scaler,
        "params": params,
        "features": list(X_train.columns)
    }

    try:
        joblib.dump(artifacts, file_path)
        logger.info("Artifacts saved at {}", file_path)
    except Exception as e:
        logger.exception("Failed to save artifacts at {}: {}", file_path, e)
        raise


def generate_reports(
    reports_path: str,
    model_name: str,
    y_true,
    y_pred,
) -> Dict[str, Any]:
    """
    Generate evaluation metrics and persist them as a JSON report.

    Args:
        reports_path: Directory where the report will be saved.
        model_name: Model identifier used in the filename.
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing evaluation metrics.
    """
    if y_true is None or y_pred is None:
        logger.error("y_true or y_pred is None")
        raise ValueError("y_true and y_pred must not be None")

    if len(y_true) == 0 or len(y_pred) == 0:
        logger.error("Empty inputs | y_true={} y_pred={}", len(y_true), len(y_pred))
        raise ValueError("y_true and y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        logger.error("Mismatched lengths | y_true={} y_pred={}", len(y_true), len(y_pred))
        raise ValueError("y_true and y_pred must have the same length")

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }

    path = Path(reports_path)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{model_name}_report.json"

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        logger.info("Report generated at {}", file_path)

    except Exception as e:
        logger.exception("Failed to generate report at {}: {}", file_path, e)
        raise

    return report
