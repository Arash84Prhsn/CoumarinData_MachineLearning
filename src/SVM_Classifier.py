import pandas as pd
import numpy as np
import json
import os

from sklearn.svm import SVC
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from sklearn.metrics import make_scorer

from tabulate import tabulate


# ============================================================
# Paths
# ============================================================

DATA_PATH = "Data/Classification_Data.csv"

RESULTS_DIR = "Results/Classification/SVM/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)


# ============================================================
# Constants
# ============================================================

RANDOM_STATE = 42
N_ITER = 100
N_SPLITS = 5

LINE = "=" * 100


# ============================================================
# Read classification data
# ============================================================

data = pd.read_csv(DATA_PATH)

print()
print(LINE)
print("Classification Data")
print(LINE)

print("Dataset shape:", data.shape)

print()
print("Columns:")
print(data.columns.to_list())


# ============================================================
# Check target distribution
# ============================================================

print()
print(LINE)
print("Target Distribution")
print(LINE)

print(data["Viability"].value_counts().sort_index())

print()
print("Target Proportions")
print(data["Viability"].value_counts(normalize=True).sort_index())


# ============================================================
# Define external test set
# ============================================================

TEST_COUMARINS = ["scopoletin", "imperatorin"]

test_mask = data["Coumarin"].isin(TEST_COUMARINS)

test_data = data[test_mask].copy()
development_data = data[~test_mask].copy()


print()
print(LINE)
print("Development / External Test Split")
print(LINE)

print("Development set:", development_data.shape)
print("External test set:", test_data.shape)

print()
print("Development coumarins:")
print(development_data["Coumarin"].value_counts())

print()
print("External test coumarins:")
print(test_data["Coumarin"].value_counts())


# ============================================================
# Features and target
# ============================================================

target = "Viability"

categorical_features = [
    "Cancer",
    "CellLine"
]

numerical_features = [
    "Dose",
    "Time",
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    "fCsp3"
]


X_development = development_data[
    categorical_features + numerical_features
]

y_development = development_data[target]

X_test = test_data[
    categorical_features + numerical_features
]

y_test = test_data[target]


# ============================================================
# Preprocessing
# ============================================================

# Categorical:
#   One-hot encoding
#
# Numerical:
#   Standardization is important for SVMs because
#   SVM optimization is sensitive to feature scale.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ============================================================
# SVM model
# ============================================================

svm = SVC(
    random_state=RANDOM_STATE
)


# ============================================================
# Pipeline
# ============================================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", svm)
    ]
)


# ============================================================
# SVM RandomizedSearchCV search space
# ============================================================

search_grid = {
    "model__C": [
        0.01,
        0.03,
        0.1,
        0.3,
        1,
        3,
        10,
        30,
        100,
        300,
        1000
    ],

    "model__kernel": [
        "linear",
        "rbf",
        "poly",
        "sigmoid"
    ],

    "model__gamma": [
        "scale",
        "auto",
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1
    ],

    "model__degree": [
        2,
        3,
        4,
        5
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]
}


# ============================================================
# Precision scorer
# ============================================================

precision_scorer = make_scorer(
    precision_score,
    zero_division=0
)


# ============================================================
# Cross-validation
# ============================================================

kfold = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# Randomized Search
# ============================================================

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=search_grid,
    n_iter=N_ITER,
    scoring=precision_scorer,
    cv=kfold,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)


# ============================================================
# Train
# ============================================================

print()
print(LINE)
print("Starting SVM RandomizedSearchCV")
print(LINE)

random_search.fit(
    X_development,
    y_development
)


# ============================================================
# Best parameters
# ============================================================

best_params = random_search.best_params_

print()
print(LINE)
print("Best Parameters")
print(LINE)

print(best_params)

print()
print("Best Cross-Validation Precision:")
print(f"{random_search.best_score_:.6f}")


# ============================================================
# Save best parameters
# ============================================================

parameters_path = (
    PARAMETERS_DIR +
    "SVM_BestParameters.json"
)

with open(parameters_path, "w") as file:
    json.dump(
        best_params,
        file,
        indent=4
    )


# ============================================================
# Reload best parameters
# ============================================================

with open(parameters_path, "r") as file:
    loaded_params = json.load(file)


# ============================================================
# Build final SVM pipeline
# ============================================================

final_svm = SVC(
    probability=True,
    random_state=RANDOM_STATE
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", final_svm)
    ]
)

final_pipeline.set_params(
    **loaded_params
)


# ============================================================
# Train final model on ALL development data
# ============================================================

print()
print(LINE)
print("Training Final SVM")
print(LINE)

final_pipeline.fit(
    X_development,
    y_development
)


# ============================================================
# External test predictions
# ============================================================

y_pred = final_pipeline.predict(X_test)

y_prob = final_pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# External test metrics
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

test_precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    y_prob
)

test_pr_auc = average_precision_score(
    y_test,
    y_prob
)


# ============================================================
# Results
# ============================================================

results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
            "PR-AUC"
        ],

        "Value": [
            test_accuracy,
            test_precision,
            test_recall,
            test_f1,
            test_roc_auc,
            test_pr_auc
        ]
    }
)


print()
print(LINE)
print("SVM External Test Results")
print(LINE)

print(
    tabulate(
        results,
        headers="keys",
        tablefmt="rounded_grid",
        showindex=False,
        floatfmt=".6f"
    )
)


# ============================================================
# Prediction distribution
# ============================================================

print()
print(LINE)
print("External Test Prediction Distribution")
print(LINE)

print(
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)


# ============================================================
# Actual target distribution
# ============================================================

print()
print(LINE)
print("External Test Actual Distribution")
print(LINE)

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# Save metrics
# ============================================================

metrics_path = RESULTS_DIR + "SVM_Metrics.csv"

results.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# Save predictions
# ============================================================

predictions = test_data[
    [
        "Author",
        "Cancer",
        "CellLine",
        "Coumarin",
        "Dose",
        "Time"
    ]
].copy()

predictions["Actual"] = y_test.values
predictions["Predicted"] = y_pred
predictions["Probability_Class_1"] = y_prob

predictions_path = RESULTS_DIR + "SVM_Test_Predictions.csv"

predictions.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# Final output
# ============================================================

print()
print(LINE)
print("Files Saved")
print(LINE)

print("Best parameters:")
print(parameters_path)

print("Metrics:")
print(metrics_path)

print("Predictions:")
print(predictions_path)

print()
print(LINE)
print("SVM Classification Complete")
print(LINE)