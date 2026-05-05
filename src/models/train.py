from pathlib import Path
import json
from typing import Dict, Any

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def train_XGBoost(X_train, y_train):
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    param_dist = {
        'n_estimators': randint(200, 1000),
        'learning_rate': [0.005, 0.01, 0.05, 0.1, 0.2],
        'max_depth': randint(3, 15),
        'min_child_weight': randint(1, 15),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 2),
    }

    xgb_clf = XGBClassifier(
        random_state=42,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss'
    )

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=200,
        cv=5,
        scoring='f1_weighted',
        refit=True,
        return_train_score=True,
        random_state=42,
        n_jobs=-1
    )

    #print("Starting hyperparameter search...")
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    #print("Finished!")

    #print(f"Best CV F1: {random_search.best_score_:.4f}")
    #print(f"Best params: {random_search.best_params_}")

    return random_search.best_estimator_, random_search.best_params_


def save_artifacts(path, model_name, model, params, scaler=None):
    artifacts = {
        "model": model,
        "scaler": scaler,
        "params": params
    }
    joblib.dump(artifacts, path + model_name)

def generate_reports(reports_path: str,
    model_name: str,
    y_true,
    y_pred
) -> Dict[str, Any]:
    """
    Gera relatório de métricas com scikit-learn e salva em JSON.

    Args:
        reports_path (str): diretório onde salvar o relatório
        model_name (str): nome do modelo
        y_true: labels reais
        y_pred: predições do modelo

    Returns:
        dict: relatório gerado
    """

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    }

    path = Path(reports_path)
    path.mkdir(parents=True, exist_ok=True)

    file_path = path / f"{model_name}_report.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return report