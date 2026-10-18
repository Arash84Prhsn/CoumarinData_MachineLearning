import pandas as pd
import numpy as np
import json
import os

from xgboost import XGBClassifier

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV
)

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    make_scorer
)


from tabulate import tabulate


# Path Constants

DATA_DIR = "Data/Processed_Data.csv"

RESULTS_DIR = "Results/Classification/"
FIGURES_DIR = RESULTS_DIR + "Figures/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"


# mkdirs commands to ensure that the directories do exist

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)


# Training constants

RANDOM_STATE = 42
N_ITER = 3000
N_SPLITS = 5


# Other constants

LINE = "=" * 100
TRIANGLE = "▶"


# Search grids for each model

xgboost_searchGrid = {

    "model__n_estimators": [
        100, 150, 200, 300, 400, 500, 600
    ],

    "model__max_depth": [
        2, 3, 4, 5, 6
    ],

    "model__learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.075, 0.1
    ],

    "model__subsample": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],

    "model__colsample_bytree": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],

    "model__min_child_weight": [
        1, 2, 3, 5, 7, 10
    ],

    "model__gamma": [
        0, 0.05, 0.1, 0.2, 0.5, 1.0
    ],

    "model__reg_alpha": [
        0, 0.001, 0.01, 0.1, 1.0
    ],

    "model__reg_lambda": [
        0.5, 1, 2, 5, 10
    ]

}


extraTrees_searchGrid = {

    "model__n_estimators": [
        200, 300, 400, 500, 700
    ],

    "model__max_depth": [
        None, 3, 4, 5, 6, 7, 8
    ],

    "model__min_samples_split": [
        2, 5, 10, 15, 20
    ],

    "model__min_samples_leaf": [
        1, 2, 4, 6, 10
    ],

    "model__max_features": [
        0.5, 0.7, 0.8, 1.0
    ],

    "model__max_leaf_nodes": [
        None, 10, 20, 30, 50, 75
    ],

    "model__bootstrap": [
        False, True
    ]

}


gradientBoosting_searchGrid = {

    "model__n_estimators": [
        100, 150, 200, 300, 400, 500
    ],

    "model__max_depth": [
        2, 3, 4, 5
    ],

    "model__learning_rate": [
        0.01, 0.02, 0.03, 0.05, 0.075, 0.1
    ],

    "model__subsample": [
        0.6, 0.7, 0.8, 0.9, 1.0
    ],

    "model__min_samples_split": [
        2, 5, 10, 15, 20
    ],

    "model__min_samples_leaf": [
        1, 2, 4, 6, 10
    ],

    "model__max_features": [
        None, "sqrt", "log2"
    ],

    "model__max_leaf_nodes": [
        None, 10, 20, 30, 50
    ]

}


randomForest_searchGrid = {

    "model__n_estimators": [
        200, 300, 400, 500, 700
    ],

    "model__max_depth": [
        None, 3, 4, 5, 6, 7, 8
    ],

    "model__min_samples_split": [
        2, 5, 10, 15, 20
    ],

    "model__min_samples_leaf": [
        1, 2, 4, 6, 10
    ],

    "model__max_features": [
        0.5, 0.7, 0.8, 1.0
    ],

    "model__max_leaf_nodes": [
        None, 10, 20, 30, 50, 75
    ],

    "model__bootstrap": [
        True, False
    ]

}


decisionTree_searchGrid = {

    "model__max_depth": [
        2, 3, 4, 5, 6
    ],

    "model__min_samples_split": [
        2, 5, 10, 15, 20, 30
    ],

    "model__min_samples_leaf": [
        1, 2, 4, 6, 10, 15
    ],

    "model__max_features": [
        None, "sqrt", "log2"
    ],

    "model__max_leaf_nodes": [
        None, 10, 15, 20, 30, 40, 50
    ],

    "model__ccp_alpha": [
        0.0, 0.0001, 0.001, 0.005, 0.01
    ]

}


# Load data

data = pd.read_csv(DATA_DIR)


# Create binary classification target
#
# Class 1: 40 < Viability < 60
# Class 0: otherwise
#
# Viability = 40 and Viability = 60 belong to class 0.

data["Viability"] = (
    (data["Viability"] > 40) &
    (data["Viability"] < 60)
).astype(int)


def tabprint(df):

    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="rounded_grid",
            showindex=False
        )
    )


print()

print(TRIANGLE, "Head of the whole data:")

tabprint(data.head())


# External test coumarins

test_coumarins = [
    "scopoletin",
    "imperatorin"
]


# Development/train-validation data

train_data = data[
    ~data["Coumarin"].isin(test_coumarins)
].drop(
    columns=["Author"]
)


# Locked external test data

test_data = data[
    data["Coumarin"].isin(test_coumarins)
].drop(
    columns=["Author"]
)


print()

print(TRIANGLE, "Train & validation data coumarins:")

print(train_data["Coumarin"].value_counts())


print()

print(TRIANGLE, "External test data coumarins:")

print(test_data["Coumarin"].value_counts())


# Feature definitions

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


feature_columns = categorical_features + numerical_features


# Classification metrics

def calculate_metrics(y_true, y_pred, y_probability):

    return {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_probability
        ),

        "PR-AUC": average_precision_score(
            y_true,
            y_probability
        )

    }


# Create model

def create_model(model_name):

    if model_name == "XGBoost":

        return XGBClassifier(

            objective="binary:logistic",

            eval_metric="logloss",

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    elif model_name == "ExtraTrees":

        return ExtraTreesClassifier(

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    elif model_name == "GradientBoosting":

        return GradientBoostingClassifier(

            random_state=RANDOM_STATE

        )

    elif model_name == "RandomForest":

        return RandomForestClassifier(

            random_state=RANDOM_STATE,

            n_jobs=-1

        )

    elif model_name == "DecisionTree":

        return DecisionTreeClassifier(

            random_state=RANDOM_STATE

        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )


precision_scorer = make_scorer(
    precision_score,
    zero_division=0
)


# Train, optimize, and externally test model

def train_and_test_model(
    model_name,
    search_grid,
    train_data,
    test_data
):

    print()

    print(LINE)

    print(
        TRIANGLE,
        f"Training {model_name}"
    )

    print(LINE)


    X = train_data[
        feature_columns
    ]

    y = train_data[
        "Viability"
    ]


    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        "Viability"
    ]


    # Preprocessing

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

                "passthrough",

                numerical_features
            )

        ]

    )


    # Create model

    model = create_model(
        model_name
    )


    # Pipeline

    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    # KFold

    kfold = KFold(

        n_splits=N_SPLITS,

        shuffle=True,

        random_state=RANDOM_STATE

    )


    # RandomizedSearchCV

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


    # Hyperparameter optimization

    random_search.fit(
        X,
        y
    )


    # Save best parameters

    parameters_path = os.path.join(

        PARAMETERS_DIR,

        f"{model_name}_BestParameters.json"

    )


    with open(
        parameters_path,
        "w"
    ) as file:

        json.dump(

            random_search.best_params_,

            file,

            indent=4

        )


    print()

    print(
        TRIANGLE,
        "Best parameters saved to:"
    )

    print(parameters_path)


    # Reload parameters

    with open(
        parameters_path,
        "r"
    ) as file:

        loaded_params = json.load(
            file
        )


    # External test predictions

    y_probability = random_search.predict_proba(
        X_test
    )[:, 1]


    y_pred = random_search.predict(
        X_test
    )


    # External test metrics

    test_metrics = calculate_metrics(

        y_test,

        y_pred,

        y_probability

    )


    print()

    print(
        TRIANGLE,
        "Best CV Precision:",
        f"{random_search.best_score_:.6f}"
    )


    print()

    print(
        TRIANGLE,
        "External test metrics:"
    )


    print(

        tabulate(

            pd.DataFrame(
                [test_metrics]
            ),

            headers="keys",

            tablefmt="rounded_grid",

            showindex=False,

            floatfmt=".6f"

        )

    )


    return {

        "model": random_search.best_estimator_,

        "best_params": loaded_params,

        "best_cv_precision": random_search.best_score_,

        "test_metrics": test_metrics,

        "y_pred": y_pred,

        "y_probability": y_probability

    }


# End of function =================================================================================>


models = {

    "DecisionTree": decisionTree_searchGrid,

    "ExtraTrees": extraTrees_searchGrid,

    "GradientBoosting": gradientBoosting_searchGrid,

    "RandomForest": randomForest_searchGrid,

    "XGBoost": xgboost_searchGrid,

}


results = {}


for model_name, search_grid in models.items():

    results[model_name] = train_and_test_model(

        model_name=model_name,

        search_grid=search_grid,

        train_data=train_data,

        test_data=test_data

    )


# Final model comparison

comparison = []


for model_name, result in results.items():

    comparison.append({

        "Model": model_name,

        "CV Precision": result["best_cv_precision"],

        "Test Accuracy": result["test_metrics"]["Accuracy"],

        "Test Precision": result["test_metrics"]["Precision"],

        "Test Recall": result["test_metrics"]["Recall"],

        "Test F1": result["test_metrics"]["F1"],

        "Test ROC-AUC": result["test_metrics"]["ROC-AUC"],

        "Test PR-AUC": result["test_metrics"]["PR-AUC"]

    })


comparison_df = pd.DataFrame(
    comparison
)


comparison_df = comparison_df.sort_values(

    "Test Precision",

    ascending=False

)


print()

print(LINE)

print(
    TRIANGLE,
    "Final model comparison"
)

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