import numpy as np
np.random.seed(42)  # All random processes are seeded with 42 for reproducibility

import pandas as pd
import os
import json
import joblib

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    train_test_split,
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

print("=" * 70)
print("STARTING XGBOOST REGRESSION SCRIPT")
print("=" * 70)

# Make the Viability feature categorical(1 if |Viability - 50| ≤ 5 and 0 otherwise)
print()
print("=" * 70)
print("TRANSFORMING VIABILITY FEATURE FOR CLASSIFICATION")
print("=" * 70)
print()

data['Viability'] = data['Viability'].apply(lambda x: 1 if abs(x - 50) <= 5 else 0)
print("Viability has been transformed by the following citeria:")
print("if |Viability - 50| ≤ 5 then set Viability to 1, set Viability to 0 otherwise")
print()

# The distribution of the classes
print("The distribution of the new clasees of Viability:")
print(tabulate(pd.DataFrame(data["Viability"].value_counts()), headers="keys", tablefmt="rounded_grid"))

# Set constants here
RANDOM_STATE = 42
RESULTS_DIR = "Results/Classification/"
MODELS_DIR = RESULTS_DIR + "Models/"
EVALUATIONS_DIR = RESULTS_DIR + "Evaluations/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"
N_ITER = 3000 # Number of iterations for the random search

# Make the required directories if they do not exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

# Take a look at the head of the data
print()
print("Head of the raw data: (Shuffled to see more variance in the data)\n")
print(tabulate(data.sample(frac=1).head(), headers='keys', tablefmt="rounded_grid"))
print()