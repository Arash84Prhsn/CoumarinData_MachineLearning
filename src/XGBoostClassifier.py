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

# Set constants here
RANDOM_STATE = 42
RESULTS_DIR = "Results/Regression/"
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
print("=" * 70)
print("STARTING XGBOOST REGRESSION SCRIPT")
print("=" * 70)

print()
print("Head of the raw data: (Shuffled to see more variance in the data)\n")
print(tabulate(data.sample(frac=1).head(), headers='keys', tablefmt="rounded_grid"))
print()