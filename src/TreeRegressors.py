import pandas as pd
import numpy as np
import json
import os

from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from tabulate import tabulate

# Path Constants
DATA_DIR = "Data/Processed_Data.csv"
RESULTS_DIR = "Results/Regression/"
FIGURES_DIR = RESULTS_DIR + "Figures/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

# mkdirs commands to ensure that the directories do exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

# Training constants
RANDOM_STATE = 42
N_ITER = 200 # For the randomsearch to find the best parameters
N_SPLITS = 5 # For KFold

# Other constants
LINE = "=" * 100
TRIANGLE = "▶"

# The search grids for each model to optimize performance:
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

gradientBoosting_searchGrid = {
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

data = pd.read_csv(DATA_DIR)

def tabprint(df):
    print(tabulate(df, headers="keys", tablefmt="rounded_grid", showindex=False))

print()
print(TRIANGLE, "Head of the whole data:")
tabprint(data.head())

test_coumarins = ["scopoletin", "imperatorin"]

train_data = data[
    ~data["Coumarin"].isin(test_coumarins)
].drop(columns=["Author"])

test_data = data[
    data["Coumarin"].isin(test_coumarins)
].drop(columns=["Author"])

print()
print(TRIANGLE, "Train & validation data coumarins:")
print(train_data["Coumarin"].value_counts())

print()
print(TRIANGLE, "External test data coumarins:")
print(test_data["Coumarin"].value_counts())

categorical_features = [
    "Cancer",
    "CellLine"
]

numerical_features = [
    "Dose",
    "Time",
    # "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    # "fCsp3"
]

feature_columns = categorical_features + numerical_features

def calculate_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def fit_encoder(train_df):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    encoder.fit(train_df[categorical_features])
    return encoder

def transform_features(df, encoder):
    X_num = df[numerical_features].to_numpy()
    X_cat = encoder.transform(df[categorical_features])
    return np.hstack([X_num, X_cat])

def create_model(model_name, parameters=None):
    if model_name == "XGBoost":
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_name == "ExtraTrees":
        model = ExtraTreesRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_name == "GradientBoosting":
        model = GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor(
            random_state=RANDOM_STATE
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if parameters is not None:
        model.set_params(**parameters)

    return model

def train_and_test_model(model_name, search_grid, train_data, test_data):
    print()
    print(LINE)
    print(TRIANGLE, f"Training {model_name}")
    print(LINE)

    X = train_data[feature_columns]
    y = train_data["Viability"]

    X_test = test_data[feature_columns]
    y_test = test_data["Viability"]

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    parameter_samples = list(
        ParameterSampler(
            search_grid,
            n_iter=N_ITER,
            random_state=RANDOM_STATE
        )
    )

    best_score = -np.inf
    best_params = None

    for iteration, parameters in enumerate(parameter_samples, start=1):
        fold_scores = []

        for train_indices, validation_indices in kfold.split(X):
            fold_train = X.iloc[train_indices]
            fold_validation = X.iloc[validation_indices]

            y_fold_train = y.iloc[train_indices]
            y_fold_validation = y.iloc[validation_indices]

            encoder = fit_encoder(fold_train)

            X_fold_train = transform_features(
                fold_train,
                encoder
            )

            X_fold_validation = transform_features(
                fold_validation,
                encoder
            )

            model = create_model(
                model_name,
                parameters
            )

            model.fit(
                X_fold_train,
                y_fold_train
            )

            y_fold_pred = model.predict(
                X_fold_validation
            )

            fold_scores.append(
                r2_score(
                    y_fold_validation,
                    y_fold_pred
                )
            )

        mean_score = np.mean(fold_scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = parameters.copy()

        if iteration % 10 == 0 or iteration == 1:
            print(
                f"Iteration {iteration:3d}/{N_ITER} | "
                f"Current R²: {mean_score:.6f} | "
                f"Best R²: {best_score:.6f}"
            )

    parameters_path = os.path.join(
        PARAMETERS_DIR,
        f"{model_name}_BestParameters.json"
    )

    with open(parameters_path, "w") as file:
        json.dump(
            best_params,
            file,
            indent=4
        )

    print()
    print(TRIANGLE, "Best parameters saved to:")
    print(parameters_path)

    with open(parameters_path, "r") as file:
        loaded_params = json.load(file)

    final_encoder = fit_encoder(X)

    X_train_final = transform_features(
        X,
        final_encoder
    )

    X_test_final = transform_features(
        X_test,
        final_encoder
    )

    final_model = create_model(
        model_name,
        loaded_params
    )

    final_model.fit(
        X_train_final,
        y
    )

    y_pred = final_model.predict(
        X_test_final
    )

    test_metrics = calculate_metrics(
        y_test,
        y_pred
    )

    print()
    print(TRIANGLE, "Best CV R²:", f"{best_score:.6f}")

    print()
    print(TRIANGLE, "External test metrics:")
    print(
        tabulate(
            pd.DataFrame([test_metrics]),
            headers="keys",
            tablefmt="rounded_grid",
            showindex=False,
            floatfmt=".6f"
        )
    )

    return {
        "model": final_model,
        "encoder": final_encoder,
        "best_params": loaded_params,
        "best_cv_r2": best_score,
        "test_metrics": test_metrics,
        "y_pred": y_pred
    };
# End of function =================================================================================>

models = {
    "DecisionTree": decisionTree_searchGrid,
    "ExtraTrees": extraTrees_searchGrid,
    "XGBoost": xgboost_searchGrid,
    "GradientBoosting": gradientBoosting_searchGrid,
    "RandomForest": randomForest_searchGrid,
}

results = {}

for model_name, search_grid in models.items():
    results[model_name] = train_and_test_model(
        model_name=model_name,
        search_grid=search_grid,
        train_data=train_data,
        test_data=test_data
    )

comparison = []

for model_name, result in results.items():
    comparison.append({
        "Model": model_name,
        "CV R2": result["best_cv_r2"],
        "Test MAE": result["test_metrics"]["MAE"],
        "Test RMSE": result["test_metrics"]["RMSE"],
        "Test MedAE": result["test_metrics"]["MedAE"],
        "Test R2": result["test_metrics"]["R2"]
    })

comparison_df = pd.DataFrame(comparison)

comparison_df = comparison_df.sort_values(
    "Test R2",
    ascending=False
)

print()
print(LINE)
print(TRIANGLE, "Final model comparison")
print(LINE)

print(
    tabulate(
        comparison_df,
        headers="keys",
        tablefmt="rounded_grid",
        showindex=False,
        floatfmt=".6f"
    )
)