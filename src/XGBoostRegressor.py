import numpy as np
np.random.seed(42)  # All random processes are seeded with 42 for reproducibility

import pandas as pd
import os
import json
import joblib

from xgboost import XGBRegressor

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    KFold,
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score
)

from tabulate import tabulate

# Load the data
data = pd.read_csv("Data/Engineered_Data.csv")

# Set constants here
RANDOM_STATE = 42
RESULTS_DIR = "Results/Regression/"
MODELS_DIR = RESULTS_DIR + "Models/"
EVALUATIONS_DIR = RESULTS_DIR + "Evaluations/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

# Make the required directories if they do not exist
os.makedirs(EVALUATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

# Take a look at the head of the data
print()
print("Head of the raw data:\n")
print(tabulate(data.head(), headers='keys', tablefmt="grid"))
print()

# Check for directories
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR);

# Set X and y
X = data.drop(columns=["Viability", "Author", "Coumarin"], inplace=False)
y = data['Viability']

print("X features:")
print(pd.Series(X.columns), end="\n\n") 

# Define the categorical and numerical features
categorical_features = ["Cancer", "CellLine"]
numerical_features = X.columns.drop(["Cancer", "CellLine"]).tolist()

# Define Groups for evaluation and parameter searching
data["GroupID"] = (
    data["Author"].astype(str) + "_" +
    data["Coumarin"].astype(str) + "_" +
    data["CellLine"].astype(str) + "_" +
    data["Time"].astype(str)
)
print("Number of uniquely identified groups using Author, Coumarin, CellLine and Time:", data["GroupID"].nunique())

print("Number Groups using only Coumarins:", data["Coumarin"].nunique())

print()

# Prepare the preprocessor
preporcessor = ColumnTransformer(
    transformers= [(
        "cat",
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        categorical_features
    ),
    (
        "num",
        "passthrough",
        numerical_features
    )]
)

# Initialize the raw Model
model = XGBRegressor(
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Define pipeline
pipeline = Pipeline([
    ("preprocessor", preporcessor),
    ("model", model)
])

# Define the search grid of the model for fine tuning the hyper parameters (Using randomSearch)
param_grid = {
    "model__n_estimators": [
        100, 150, 200, 300, 400, 500,
        600, 800, 1000
    ],

    "model__max_depth": [
        2, 3, 4, 5, 6, 7, 8
    ],

    "model__learning_rate": [
        0.005, 0.01, 0.02, 0.03,
        0.05, 0.075, 0.1, 0.15, 0.2
    ],

    "model__subsample": [
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0
    ],

    "model__colsample_bytree": [
        0.5, 0.6, 0.7, 0.8,
        0.9, 1.0
    ],

    "model__min_child_weight": [
        1, 2, 3, 5, 7, 10
    ],

    "model__gamma": [
        0, 0.01, 0.05, 0.1,
        0.2, 0.5, 1.0
    ],

    "model__reg_alpha": [
        0, 0.001, 0.01, 0.1, 1.0
    ],

    "model__reg_lambda": [
        0.1, 0.5, 1, 2, 5, 10
    ]
}

# Next up we'll define some helpers so that reproducubility under different conditions is simpler

# Helper function to get the results of evaluation
def calculate_metrics(y_true, y_pred):
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE" : np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE" : median_absolute_error(y_true, y_pred),
        "R2" : r2_score(y_true, y_pred)
    };

# Evaluation methods:
eval_methods = {
    "Kfold" : {
        "splitter" : KFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        ),
        "groups" : None
    },

    "GroupID" : {
        "splitter" : GroupKFold(
            n_splits=5
        ),
        "groups" : data["GroupID"]
    },

    "Coumarin" : {
        "splitter" : GroupKFold(
            n_splits=data["Coumarin"].nunique()
        ),
        "groups" : data["Coumarin"]
    }
}

# We can now start the evaluation process
eval_results = []

for method_name, method in eval_methods.items():

    print()
    print("=" * 70)
    print(f"Evaluation method: {method_name}")
    print("=" * 70)
    print()

    splitter = method["splitter"]
    groups = method["groups"]

    if method_name == "Kfold":
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=0.20,
            random_state=RANDOM_STATE
        )