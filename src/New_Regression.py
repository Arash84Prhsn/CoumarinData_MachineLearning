import pandas as pd;
import numpy as np;
np.random.seed(42);
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from tabulate import tabulate
import json
import os

# Define path Constants
DATA_DIR = "Data/Processed_Data.csv"
RESULTS_DIR = "Results/Regression/"
FIGURES_DIR = RESULTS_DIR + "Figures/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

# mkdirs just in case
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

# Define other Constants
RANDOM_STATE = 42
N_ITER = 100
N_SPLITS = 5
LINE = "=" * 100 + "\n"
TRIANGLE = "▶"

# Load Data
data = pd.read_csv(DATA_DIR);

# Look at data
def tabprint(df) :
    print(tabulate(df, headers='keys'));

print()
print(TRIANGLE, "Head of the whole data: \n")
tabprint(data.head())

# Define the coumarins that will be used for training and validation and the coumarins for final testing of model
train_data = data[~data["Coumarin"].isin(["scopoletin", "imperatorin"])].drop(columns=['Author'])
test_data = data[data["Coumarin"].isin(["scopoletin", "imperatorin"])].drop(columns=['Author'])

print()
print(TRIANGLE, "Train & validation data coumarins:")   
print(train_data["Coumarin"].value_counts())

print()
print(TRIANGLE, "Test data coumarins:")
print(test_data["Coumarin"].value_counts())

# Prepare to transform the categorical features and scale the numerical features:
categorical_features = ["Cancer", "CellLine"]

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

# Define Search grids for the models: (Random search will be done)

xgboost_searchGrid = {
    "n_estimators": [100, 150, 200, 300, 400, 500, 600],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "gamma": [0, 0.05, 0.1, 0.2, 0.5, 1.0],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1, 2, 5, 10]
}

extraTrees_searchGrid = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [None, 3, 4, 5, 6, 7, 8],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 10],
    "max_features": [0.5, 0.7, 0.8, 1.0],
    "max_leaf_nodes": [None, 10, 20, 30, 50, 75],
    "bootstrap": [False, True]
}

gradientBoosting_searhGrid = {
    "n_estimators": [100, 150, 200, 300, 400, 500],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 10],
    "max_features": [None, "sqrt", "log2"],
    "max_leaf_nodes": [None, 10, 20, 30, 50]
}

randomForest_searchGrid = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [None, 3, 4, 5, 6, 7, 8],
    "min_samples_split": [2, 5, 10, 15, 20],
    "min_samples_leaf": [1, 2, 4, 6, 10],
    "max_features": [0.5, 0.7, 0.8, 1.0],
    "max_leaf_nodes": [None, 10, 20, 30, 50, 75],
    "bootstrap": [True, False]
}

decisionTree_searchGrid = {
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_split": [2, 5, 10, 15, 20, 30],
    "min_samples_leaf": [1, 2, 4, 6, 10, 15],
    "max_features": [None, "sqrt", "log2"],
    "max_leaf_nodes": [None, 10, 15, 20, 30, 40, 50],
    "ccp_alpha": [0.0, 0.0001, 0.001, 0.005, 0.01]
}

# Functions:
# Get the metrics
def calculate_metrics(y_true, y_pred):
    """Get the metric scores MAE, RMSE, MedAE, R2 in a dict"""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    };

# train and test function:
def train_and_test_model(model_name, model, search_grid, X_train, y_train, X_test, y_test):
    print()
    print(TRIANGLE, f"Training {model_name}...")

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_grid,
        n_iter=N_ITER,
        scoring="r2",
        cv=kfold,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_
    best_cv_score = random_search.best_score_

    print()
    print(TRIANGLE, f"Best CV R² for {model_name}: {best_cv_score:.6f}")
    print(TRIANGLE, "Best parameters:")
    print(json.dumps(best_params, indent=4))

    # Save best parameters
    parameters_path = os.path.join(
        PARAMETERS_DIR,
        f"{model_name}_BestParameters.json"
    )

    with open(parameters_path, "w") as file:
        json.dump(best_params, file, indent=4)

    # Load best parameters back from JSON
    with open(parameters_path, "r") as file:
        loaded_params = json.load(file)

    # Train final model on all training/validation data
    final_model = model.__class__(
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    final_model.set_params(**loaded_params)
    final_model.fit(X_train, y_train)

    # External test
    y_pred = final_model.predict(X_test)

    test_metrics = calculate_metrics(y_test, y_pred)

    print()
    print(TRIANGLE, f"External test results for {model_name}:")
    print(tabulate(
        pd.DataFrame([test_metrics]),
        headers="keys",
        tablefmt="rounded_grid",
        showindex=False,
        floatfmt=".6f"
    ))

    return {
        "model": final_model,
        "best_params": loaded_params,
        "best_cv_r2": best_cv_score,
        "test_metrics": test_metrics,
        "y_pred": y_pred
    };

X_train = train_data.drop(columns=["Viability", "Coumarin"])
y_train = train_data["Viability"]

X_test = test_data.drop(columns=["Viability", "Coumarin"])
y_test = test_data["Viability"]