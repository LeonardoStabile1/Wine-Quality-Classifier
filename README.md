# Wine Quality Classifier

An end-to-end machine learning project for wine quality classification using physicochemical attributes from red and white wines.



![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC1C24?logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?logo=pytest&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white)



## Overview

This project was developed to classify wine quality categories using supervised machine learning techniques. The pipeline covers the complete workflow, including data preprocessing, feature engineering, model training, evaluation, and API deployment.

The repository was structured following software engineering and MLOps best practices, with focus on modularity, reproducibility, testing, and production-ready organization.

To see the instructions on how to install and run this project, check the [running_this_project.md](running_this_project.md).


## Features

This project covers the complete machine learning workflow, from data preprocessing and feature engineering to model training, evaluation, and deployment. The model was built and optimized using XGBoost, and the application exposes predictions through a FastAPI service. The repository follows a modular structure to improve maintainability and reproducibility, includes automated tests with Pytest, and stores trained models, reports, and other experiment artifacts generated during development.


## Model Performance

The final model is an optimized XGBoost classifier trained to predict wine quality categories from physicochemical measurements. After preprocessing and feature engineering, the model was tuned to balance predictive performance with inference efficiency, making it suitable for real-time API deployment.

The evaluation results show stable generalization across unseen validation data, with an F1-score of 81.68%, precision of 81.34%, and recall of 82.15%. These metrics indicate that the model maintains a good balance between minimizing false predictions and correctly identifying wine quality classes.

| Metric | Score |
|---|---|
| Precision | 81.34% |
| Recall | 82.15% |
| F1-Score | 81.68% |

The model achieved solid generalization performance while maintaining a lightweight inference pipeline suitable for real-time predictions.


## Dataset

The dataset contains physicochemical measurements from red and white wines, including features such as acidity, alcohol concentration, pH, sulphates, chlorides, and residual sugar.

Dataset source:  
[Wine Quality Dataset (Kaggle)](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset?utm_source=chatgpt.com)



## Project Structure

```text
.
├── dataset/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── api/
│   ├── config/
│   ├── data/
│   ├── models/
│   └── pipelines/
├── tests/
├── Makefile
├── requirements.txt
├── running_this_project.md
└── README.md
```

## API

The project also includes a FastAPI application that serves the trained model through a REST API, allowing real-time wine quality predictions from structured input data. The API was designed to provide a lightweight and production-oriented inference layer, making it easier to integrate the model into external applications or services.

In addition to prediction endpoints, the service exposes health and readiness checks to monitor application status and verify that the model is properly loaded before inference requests are processed. A dedicated endpoint is also available to retrieve model metadata and expected input features.

### Main Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Application health check |
| GET | `/ready` | Model readiness check |
| POST | `/predict` | Generate predictions |
| GET | `/model-info` | Model metadata and features |

### Example Prediction Request

```json
{
  "data": [
    {
      "fixed acidity": 0,
      "volatile acidity": 0,
      "citric acid": 0,
      "residual sugar": 0,
      "chlorides": 0,
      "free sulfur dioxide": 0,
      "total sulfur dioxide": 0,
      "density": 0,
      "pH": 0,
      "sulphates": 0,
      "alcohol": 0,
      "type": "red"
    }
  ]
}
```

---

## License

This project was developed for educational, research, and portfolio purposes.