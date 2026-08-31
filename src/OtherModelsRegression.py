import numpy as np

np.random.seed(42)

import pandas as pd
import os
import json
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
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

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.linear_model import (
    Ridge,
    ElasticNet
)

from sklearn.svm import SVR

from scipy.stats import loguniform

from tabulate import tabulate

import warnings

warnings.filterwarnings("ignore")


# ==================================================================================================
# CONSTANTS
# ==================================================================================================

RANDOM_STATE = 42

N_ITER = 300

DATA_PATH = "Data/Engineered_Data.csv"

RESULTS_DIR = "Results/Regression/OtherModels/"

MODELS_DIR = RESULTS_DIR + "Models/"
EVALUATIONS_DIR = RESULTS_DIR + "Evaluations/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"


# ==================================================================================================
# CREATE DIRECTORIES
# ==================================================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)


# ==================================================================================================
# LOAD DATA
# ==================================================================================================

data = pd.read_csv(DATA_PATH)


print("=" * 70)
print("STARTING MULTI-MODEL REGRESSION SCRIPT")
print("=" * 70)
print()

print("Head of the raw data:\n")

print(
    tabulate(
        data.sample(frac=1).head(),
        headers="keys",
        tablefmt="rounded_grid"
    )
)

print()


# ==================================================================================================
# DEFINE X AND y
# ==================================================================================================

print("=" * 70)
print("DEFINING X AND y")
print("=" * 70)
print()


X = data.drop(
    columns=[
        "Viability",
        "Author",
        "Coumarin"
    ],
    inplace=False
)

y = data["Viability"]


print("X features:")

print(
    tabulate(
        pd.DataFrame(X.columns),
        tablefmt="rounded_grid"
    )
)

print()

print("The y vector has been set to the `Viability` feature.")


# ==================================================================================================
# FEATURE TYPES
# ==================================================================================================

categorical_features = [
    "Cancer",
    "CellLine"
]

numerical_features = X.columns.drop(
    categorical_features
).tolist()


# ==================================================================================================
# DEFINE GROUPS
# ==================================================================================================

data["GroupID"] = (
    data["Author"].astype(str)
    + "_"
    + data["Coumarin"].astype(str)
    + "_"
    + data["CellLine"].astype(str)
    + "_"
    + data["Time"].astype(str)
)


print()

print("=" * 70)
print("DEFINING GROUPING STRATEGIES")
print("=" * 70)
print()

print(
    "1. Grouping using Author, Coumarin, CellLine and Time"
)

print(
    "Number of uniquely identified groups:",
    data["GroupID"].nunique()
)

print()

print(
    "2. Grouping using Coumarin"
)

print(
    "Number of Coumarin groups:",
    data["Coumarin"].nunique()
)

print()

print(
    "3. Standard 5-fold KFold"
)


# ==================================================================================================
# EVALUATION METHODS
# ==================================================================================================

eval_methods = {

    "Kfold": {

        "splitter": KFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE
        ),

        "groups": None
    },

    "GroupID": {

        "splitter": GroupKFold(
            n_splits=5
        ),

        "groups": data["GroupID"]
    },

    "Coumarin": {

        "splitter": GroupKFold(
            n_splits=data["Coumarin"].nunique()
        ),

        "groups": data["Coumarin"]
    }
}


# ==================================================================================================
# PREPROCESSORS
# ==================================================================================================

def create_tree_preprocessor():

    return ColumnTransformer(

        transformers=[

            (
                "cat",

                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                ),

                categorical_features
            ),

            (
                "num",
                "passthrough",
                numerical_features
            )
        ]
    )


def create_scaled_preprocessor():

    return ColumnTransformer(

        transformers=[

            (
                "cat",

                OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                ),

                categorical_features
            ),

            (
                "num",

                StandardScaler(),

                numerical_features
            )
        ]
    )


# ==================================================================================================
# MODEL DEFINITIONS
# ==================================================================================================

models = {

    "RandomForest": {

        "model": RandomForestRegressor(
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "preprocessor": create_tree_preprocessor(),

        "params": {

            "model__n_estimators": [
                100,
                200,
                300,
                500,
                800,
                1000
            ],

            "model__max_depth": [
                None,
                3,
                5,
                7,
                10,
                15,
                20,
                30
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                15,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4,
                8,
                12
            ],

            "model__max_features": [
                0.5,
                0.7,
                1.0
            ]
        }
    },


    "ExtraTrees": {

        "model": ExtraTreesRegressor(
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "preprocessor": create_tree_preprocessor(),

        "params": {

            "model__n_estimators": [
                100,
                200,
                300,
                500,
                800,
                1000
            ],

            "model__max_depth": [
                None,
                3,
                5,
                7,
                10,
                15,
                20,
                30
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                15,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4,
                8,
                12
            ],

            "model__max_features": [
                0.5,
                0.7,
                1.0
            ]
        }
    },


    "GradientBoosting": {

        "model": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "preprocessor": create_tree_preprocessor(),

        "params": {

            "model__n_estimators": [
                100,
                150,
                200,
                300,
                500,
                800
            ],

            "model__max_depth": [
                2,
                3,
                4,
                5,
                6,
                8
            ],

            "model__learning_rate": [
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2
            ],

            "model__subsample": [
                0.5,
                0.7,
                0.8,
                0.9,
                1.0
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                15,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4,
                8,
                12
            ],

            "model__max_features": [
                None,
                "sqrt",
                "log2"
            ]
        }
    },


    "HistGradientBoosting": {

        "model": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),

        "preprocessor": create_tree_preprocessor(),

        "params": {

            "model__max_iter": [
                100,
                150,
                200,
                300,
                500,
                800
            ],

            "model__learning_rate": [
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2
            ],

            "model__max_leaf_nodes": [
                7,
                15,
                31,
                63,
                127
            ],

            "model__max_depth": [
                None,
                3,
                5,
                7,
                10
            ],

            "model__min_samples_leaf": [
                5,
                10,
                15,
                20,
                30
            ],

            "model__l2_regularization": [
                0,
                0.01,
                0.1,
                1,
                10
            ]
        }
    },


    "Ridge": {

        "model": Ridge(),

        "preprocessor": create_scaled_preprocessor(),

        "params": {

            "model__alpha": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000
            ]
        }
    },


    "ElasticNet": {

        "model": ElasticNet(
            max_iter=10000,
            random_state=RANDOM_STATE
        ),

        "preprocessor": create_scaled_preprocessor(),

        "params": {

            "model__alpha": [
                0.0001,
                0.001,
                0.01,
                0.1,
                1,
                10,
                100
            ],

            "model__l1_ratio": [
                0.0,
                0.1,
                0.25,
                0.5,
                0.75,
                0.9,
                1.0
            ]
        }
    },


    "SVR": {

        "model": SVR(),

        "preprocessor": create_scaled_preprocessor(),

        "params": {

            "model__C": loguniform(
                0.01,
                1000
            ),

            "model__gamma": [
                "scale",
                "auto"
            ],

            "model__epsilon": [
                0.01,
                0.05,
                0.1,
                0.2,
                0.3,
                0.5,
                1.0
            ],

            "model__kernel": [
                "rbf"
            ]
        }
    }
}


# ==================================================================================================
# HELPER: CREATE MODEL
# ==================================================================================================

def create_model(model_name):

    model_info = models[model_name]

    return Pipeline([

        (
            "preprocessor",
            model_info["preprocessor"]
        ),

        (
            "model",
            model_info["model"]
        )
    ])


# ==================================================================================================
# HELPER: CREATE MODEL WITH PARAMETERS
# ==================================================================================================

def create_model_with_params(
    model_name,
    best_params
):

    model_pipeline = create_model(
        model_name
    )

    model_pipeline.set_params(
        **best_params
    )

    return model_pipeline


# ==================================================================================================
# METRICS
# ==================================================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "MAE":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "MedAE":
            median_absolute_error(
                y_true,
                y_pred
            ),

        "R2":
            r2_score(
                y_true,
                y_pred
            )
    }


# ==================================================================================================
# HYPERPARAMETER SEARCH
# ==================================================================================================

def search_best_parameters(
    model_name,
    method_name,
    splitter,
    groups
):

    parameter_path = (

        PARAMETERS_DIR
        + f"{model_name}_{method_name}.json"
    )


    print()
    print("=" * 70)

    print(
        f"Starting hyperparameter search:"
        f" {model_name} - {method_name}"
    )

    print("=" * 70)
    print()


    # ------------------------------------------------
    # Check existing parameters
    # ------------------------------------------------

    if os.path.exists(parameter_path):

        print(
            f"[{model_name} - {method_name}] "
            "Best parameters already exist."
        )

        print(
            "Skipping hyperparameter search."
        )

        with open(
            parameter_path,
            "r"
        ) as file:

            best_params = json.load(file)

        return best_params


    # ------------------------------------------------
    # Create pipeline
    # ------------------------------------------------

    pipeline = create_model(
        model_name
    )


    # ------------------------------------------------
    # Create search
    # ------------------------------------------------

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=models[
            model_name
        ]["params"],

        n_iter=N_ITER,

        scoring="r2",

        cv=splitter,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=False,

        verbose=1
    )


    # ------------------------------------------------
    # Fit search
    # ------------------------------------------------

    if groups is None:

        search.fit(
            X,
            y
        )

    else:

        search.fit(
            X,
            y,
            groups=groups
        )


    # ------------------------------------------------
    # Get best parameters
    # ------------------------------------------------

    best_params = search.best_params_


    print()

    print(
        f"Best CV R2 for "
        f"{model_name} - {method_name}: "
        f"{search.best_score_:.6f}"
    )

    print()

    print("Best parameters:")

    for parameter, value in best_params.items():

        print(
            f"  {parameter}: {value}"
        )


    # ------------------------------------------------
    # Save parameters
    # ------------------------------------------------

    with open(
        parameter_path,
        "w"
    ) as file:

        json.dump(
            best_params,
            file,
            indent=4
        )


    print()

    print(
        "Best parameters saved to:"
    )

    print(
        f"  {parameter_path}"
    )


    return best_params


# ==================================================================================================
# MODEL EVALUATION
# ==================================================================================================

def evaluate_model(
    model_name,
    method_name,
    splitter,
    groups,
    best_params
):

    print()

    print("=" * 70)

    print(
        f"Evaluating {model_name}: "
        f"using {method_name} splitting method"
    )

    print("=" * 70)

    print()


    fold_results = []


    # ------------------------------------------------
    # Generate CV splits
    # ------------------------------------------------

    if groups is None:

        split_iterator = splitter.split(
            X,
            y
        )

    else:

        split_iterator = splitter.split(
            X,
            y,
            groups=groups
        )


    # ------------------------------------------------
    # Evaluate folds
    # ------------------------------------------------

    for fold_number, (
        train_idx,
        test_idx
    ) in enumerate(
        split_iterator,
        start=1
    ):

        print(
            f"Evaluating fold {fold_number}..."
        )


        # ------------------------------------------------
        # Identify left-out coumarin
        # ------------------------------------------------

        left_out_coumarin = None


        if method_name == "Coumarin":

            left_out_coumarin = str(
                data.iloc[test_idx]["Coumarin"].iloc[0]
            )

            print(
                "Left Out Coumarin:",
                left_out_coumarin
            )


        # ------------------------------------------------
        # Split data
        # ------------------------------------------------

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]


        # ------------------------------------------------
        # Fresh model
        # ------------------------------------------------

        fold_model = create_model_with_params(

            model_name,

            best_params
        )


        # ------------------------------------------------
        # Train
        # ------------------------------------------------

        fold_model.fit(
            X_train,
            y_train
        )


        # ------------------------------------------------
        # Predict
        # ------------------------------------------------

        y_pred = fold_model.predict(
            X_test
        )


        # ------------------------------------------------
        # Calculate metrics
        # ------------------------------------------------

        metrics = calculate_metrics(
            y_test,
            y_pred
        )


        fold_results.append({

            "Model":
                model_name,

            "Method":
                method_name,

            "Fold":
                fold_number,

            "Left Out Coumarin":
                left_out_coumarin,

            **metrics
        })


        print(
            f"  R2:    {metrics['R2']:.6f}"
        )

        print(
            f"  MAE:   {metrics['MAE']:.6f}"
        )

        print(
            f"  RMSE:  {metrics['RMSE']:.6f}"
        )

        print(
            f"  MedAE: {metrics['MedAE']:.6f}"
        )

        print()


    # ------------------------------------------------
    # DataFrame
    # ------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )


    # ------------------------------------------------
    # Calculate summary
    # ------------------------------------------------

    metric_names = [
        "MAE",
        "RMSE",
        "MedAE",
        "R2"
    ]


    summary = {}


    for metric in metric_names:

        summary[
            f"{metric}_Mean"
        ] = fold_results_df[
            metric
        ].mean()


        summary[
            f"{metric}_Std"
        ] = fold_results_df[
            metric
        ].std()


    return (
        fold_results_df,
        summary
    )


# ==================================================================================================
# MAIN TRAINING / EVALUATION LOOP
# ==================================================================================================

all_fold_results = []

all_summary_results = []


for model_name in models:

    print()

    print("#" * 70)

    print(
        f"PROCESSING MODEL: {model_name}"
    )

    print("#" * 70)


    for method_name, method in eval_methods.items():

        splitter = method["splitter"]

        groups = method["groups"]


        # ------------------------------------------------
        # Hyperparameter search
        # ------------------------------------------------

        best_params = search_best_parameters(

            model_name=model_name,

            method_name=method_name,

            splitter=splitter,

            groups=groups
        )


        # ------------------------------------------------
        # Evaluation
        # ------------------------------------------------

        fold_results_df, summary = evaluate_model(

            model_name=model_name,

            method_name=method_name,

            splitter=splitter,

            groups=groups,

            best_params=best_params
        )


        # ------------------------------------------------
        # Store results
        # ------------------------------------------------

        all_fold_results.append(
            fold_results_df
        )


        all_summary_results.append({

            "Model":
                model_name,

            "Method":
                method_name,

            **summary
        })


# ==================================================================================================
# COMBINE RESULTS
# ==================================================================================================

all_fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)


summary_df = pd.DataFrame(
    all_summary_results
)


# ==================================================================================================
# PRINT FOLD RESULTS
# ==================================================================================================

print()

print("=" * 70)

print(
    "FOLD-LEVEL EVALUATION RESULTS"
)

print("=" * 70)

print()


print(
    tabulate(
        all_fold_results_df,
        headers="keys",
        tablefmt="rounded_grid",
        showindex=False,
        floatfmt=".6f"
    )
)


# ==================================================================================================
# PRINT SUMMARY
# ==================================================================================================

print()

print("=" * 70)

print(
    "EVALUATION SUMMARY"
)

print("=" * 70)

print()


print(
    tabulate(
        summary_df,
        headers="keys",
        tablefmt="rounded_grid",
        showindex=False,
        floatfmt=".6f"
    )
)


# ==================================================================================================
# SAVE EVALUATIONS
# ==================================================================================================

all_fold_results_df.to_csv(

    EVALUATIONS_DIR
    + "AllModels_FoldResults.csv",

    index=False
)


summary_df.to_csv(

    EVALUATIONS_DIR
    + "AllModels_EvaluationSummary.csv",

    index=False
)


# ==================================================================================================
# TRAIN FINAL MODELS
# ==================================================================================================

print()

print("=" * 70)

print(
    "TRAINING FINAL MODELS"
)

print("=" * 70)

print()


for model_name in models:

    for method_name in eval_methods:

        parameter_path = (

            PARAMETERS_DIR
            + f"{model_name}_{method_name}.json"
        )


        model_path = (

            MODELS_DIR
            + f"{model_name}_{method_name}.joblib"
        )


        # ------------------------------------------------
        # Load best parameters
        # ------------------------------------------------

        with open(
            parameter_path,
            "r"
        ) as file:

            best_params = json.load(
                file
            )


        # ------------------------------------------------
        # Create model
        # ------------------------------------------------

        final_model = create_model_with_params(

            model_name,

            best_params
        )


        # ------------------------------------------------
        # Train on all data
        # ------------------------------------------------

        final_model.fit(
            X,
            y
        )


        # ------------------------------------------------
        # Save model
        # ------------------------------------------------

        joblib.dump(
            final_model,
            model_path
        )


        print(
            f"{model_name} - {method_name} "
            "final model saved to:"
        )

        print(
            f"  {model_path}"
        )


print()

print("=" * 70)

print(
    "ALL MODELS FINISHED SUCCESSFULLY!"
)

print("=" * 70)

print()