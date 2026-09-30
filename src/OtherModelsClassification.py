import numpy as np

np.random.seed(42)

import pandas as pd
import os
import json
import joblib

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RandomizedSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.svm import SVC

from tabulate import tabulate


# ===================================================================================================
# LOAD DATA
# ===================================================================================================

data = pd.read_csv(
    "Data/Engineered_Data.csv"
)

print("=" * 70)
print("STARTING MULTI-MODEL CLASSIFICATION SCRIPT")
print("=" * 70)


# ===================================================================================================
# TRANSFORM VIABILITY INTO CLASSIFICATION TARGET
# ===================================================================================================

print()
print("=" * 70)
print("TRANSFORMING VIABILITY FEATURE FOR CLASSIFICATION")
print("=" * 70)
print()

data["Target"] = (
    (data["Viability"] - 50).abs() <= 5
).astype(int)

print(
    "Target has been transformed using the following criteria:"
)

print(
    "if |Viability - 50| <= 5 then Target = 1, otherwise Target = 0"
)

print()

print("Distribution of the new classes:")

print(
    tabulate(
        pd.DataFrame(
            data["Target"].value_counts()
        ),
        headers="keys",
        tablefmt="rounded_grid"
    )
)


# ===================================================================================================
# CONSTANTS
# ===================================================================================================

RANDOM_STATE = 42

RESULTS_DIR = "Results/Classification/OtherModels/"

MODELS_DIR = RESULTS_DIR + "Models/"
EVALUATIONS_DIR = RESULTS_DIR + "Evaluations/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"

N_ITER = 50


# ===================================================================================================
# CREATE DIRECTORIES
# ===================================================================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    EVALUATIONS_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    PARAMETERS_DIR,
    exist_ok=True
)


# ===================================================================================================
# DISPLAY DATA
# ===================================================================================================

print()

print(
    "Head of the raw data:"
)

print()

print(
    tabulate(
        data.sample(frac=1).head(),
        headers="keys",
        tablefmt="rounded_grid"
    )
)


# ===================================================================================================
# DEFINE X AND y
# ===================================================================================================

print()
print("=" * 70)
print("DEFINING X AND y")
print("=" * 70)
print()

X = data.drop(
    columns=[
        "Target",
        "Viability",
        "Author",
        "Coumarin"
    ],
    inplace=False
)

y = data["Target"]

print("X features:")

print(
    tabulate(
        pd.DataFrame(X.columns),
        tablefmt="rounded_grid"
    )
)

print()

print(
    "The y vector has been set to the Target feature."
)


# ===================================================================================================
# FEATURE TYPES
# ===================================================================================================

categorical_features = [
    "Cancer",
    "CellLine"
]

numerical_features = X.columns.drop(
    categorical_features
).tolist()


# ===================================================================================================
# DEFINE GROUPS
# ===================================================================================================

data["GroupID"] = (
    data["Author"].astype(str) + "_" +
    data["Coumarin"].astype(str) + "_" +
    data["CellLine"].astype(str) + "_" +
    data["Time"].astype(str)
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
    "Number of unique groups:",
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
    "3. Standard KFold using 5 folds"
)


# ===================================================================================================
# EVALUATION METHODS
# ===================================================================================================

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


# ===================================================================================================
# MODEL DEFINITIONS
# ===================================================================================================

models = {

    "LogisticRegression": {

        "model": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),

        "scale": True,

        "params": {

            "model__C": [
                0.001,
                0.01,
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                50,
                100
            ],

            "model__solver": [
                "lbfgs",
                "liblinear"
            ],

            "model__class_weight": [
                None,
                "balanced"
            ]
        }
    },


    "RandomForest": {

        "model": RandomForestClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "scale": False,

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
                8,
                10,
                15,
                20,
                30
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                5,
                10
            ],

            "model__max_features": [
                "sqrt",
                "log2",
                0.5,
                0.75,
                1.0
            ],

            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample"
            ]
        }
    },


    "ExtraTrees": {

        "model": ExtraTreesClassifier(
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "scale": False,

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
                8,
                10,
                15,
                20,
                30
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                5,
                10
            ],

            "model__max_features": [
                "sqrt",
                "log2",
                0.5,
                0.75,
                1.0
            ],

            "model__class_weight": [
                None,
                "balanced",
                "balanced_subsample"
            ]
        }
    },


    "GradientBoosting": {

        "model": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "scale": False,

        "params": {

            "model__n_estimators": [
                50,
                100,
                150,
                200,
                300,
                500
            ],

            "model__learning_rate": [
                0.01,
                0.02,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2
            ],

            "model__max_depth": [
                1,
                2,
                3,
                4,
                5,
                6
            ],

            "model__min_samples_split": [
                2,
                5,
                10,
                20
            ],

            "model__min_samples_leaf": [
                1,
                2,
                5,
                10
            ],

            "model__subsample": [
                0.5,
                0.7,
                0.8,
                0.9,
                1.0
            ]
        }
    },


    "HistGradientBoosting": {

        "model": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),

        "scale": False,

        "params": {

            "model__max_iter": [
                50,
                100,
                150,
                200,
                300,
                500
            ],

            "model__learning_rate": [
                0.01,
                0.02,
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
                10,
                20,
                30,
                50
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


    "SVC": {

        "model": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "scale": True,

        "params": {

            "model__C": [
                0.001,
                0.01,
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                50,
                100
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
                0.001,
                0.01,
                0.05,
                0.1,
                0.5,
                1
            ],

            "model__class_weight": [
                None,
                "balanced"
            ]
        }
    }

}


# ===================================================================================================
# CREATE PREPROCESSOR
# ===================================================================================================

def create_preprocessor(scale):

    if scale:

        numerical_transformer = Pipeline([
            (
                "scaler",
                StandardScaler()
            )
        ])

    else:

        numerical_transformer = "passthrough"


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

                numerical_transformer,

                numerical_features
            )
        ]
    )


# ===================================================================================================
# CREATE MODEL PIPELINE
# ===================================================================================================

def create_model(model_name):

    model_information = models[model_name]

    return Pipeline([

        (
            "preprocessor",

            create_preprocessor(
                model_information["scale"]
            )
        ),

        (
            "model",

            model_information["model"]
        )
    ])


# ===================================================================================================
# CREATE MODEL WITH PARAMETERS
# ===================================================================================================

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


# ===================================================================================================
# CALCULATE CLASSIFICATION METRICS
# ===================================================================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    metrics = {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "BalancedAccuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "ROC_AUC":
            np.nan,

        "PR_AUC":
            average_precision_score(
                y_true,
                y_prob
            )
    }


    if len(np.unique(y_true)) == 2:

        metrics["ROC_AUC"] = roc_auc_score(
            y_true,
            y_prob
        )


    return metrics


# ===================================================================================================
# HYPERPARAMETER SEARCH
# ===================================================================================================

def search_best_parameters(
    model_name,
    method_name,
    splitter,
    groups
):

    parameter_path = (

        PARAMETERS_DIR +

        f"{model_name}_{method_name}.json"
    )


    print()
    print("=" * 70)

    print(
        f"Starting hyperparameter search:"
        f" {model_name} - {method_name}"
    )

    print("=" * 70)
    print()


    # --------------------------------------------------------
    # Check whether parameters already exist
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    pipeline = create_model(
        model_name
    )


    # --------------------------------------------------------
    # Get parameter distribution
    # --------------------------------------------------------

    param_distributions = models[
        model_name
    ]["params"]


    # --------------------------------------------------------
    # Randomized search
    # --------------------------------------------------------

    search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_distributions,

        n_iter=N_ITER,

        scoring="roc_auc",

        cv=splitter,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        refit=False,

        verbose=1
    )


    # --------------------------------------------------------
    # Fit search
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Retrieve best parameters
    # --------------------------------------------------------

    best_params = search.best_params_


    print()

    print(
        f"Best CV ROC-AUC for "
        f"{model_name} / {method_name}: "
        f"{search.best_score_:.6f}"
    )

    print()

    print("Best parameters:")

    for parameter, value in best_params.items():

        print(
            f"  {parameter}: {value}"
        )


    # --------------------------------------------------------
    # Save parameters
    # --------------------------------------------------------

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


# ===================================================================================================
# EVALUATE MODEL
# ===================================================================================================

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
        f"Evaluating {model_name}"
    )

    print(
        f"Splitting method: {method_name}"
    )

    print("=" * 70)

    print()


    fold_results = []


    # --------------------------------------------------------
    # Generate CV splits
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Evaluate every fold
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # Identify left-out coumarin
        # ----------------------------------------------------

        left_out_coumarin = None

        if method_name == "Coumarin":

            left_out_coumarin = str(
                data.iloc[test_idx]["Coumarin"].iloc[0]
            )

            print(
                "Left Out Coumarin:",
                left_out_coumarin
            )


        # ----------------------------------------------------
        # Split data
        # ----------------------------------------------------

        X_train = X.iloc[train_idx]

        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        y_test = y.iloc[test_idx]


        # ----------------------------------------------------
        # Fresh model for this fold
        # ----------------------------------------------------

        fold_model = create_model_with_params(

            model_name,

            best_params
        )


        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        fold_model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = fold_model.predict(
            X_test
        )

        y_prob = fold_model.predict_proba(
            X_test
        )[:, 1]


        # ----------------------------------------------------
        # Print distributions
        # ----------------------------------------------------

        print(
            "Actual class distribution:"
        )

        print(
            pd.Series(
                y_test
            ).value_counts().to_dict()
        )


        print(
            "Predicted class distribution:"
        )

        print(
            pd.Series(
                y_pred
            ).value_counts().to_dict()
        )


        print(
            f"Predicted probability range: "
            f"{y_prob.min():.4f} - "
            f"{y_prob.max():.4f}"
        )


        print(
            f"Mean predicted probability: "
            f"{y_prob.mean():.4f}"
        )


        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(

            y_test,

            y_pred,

            y_prob
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


        # ----------------------------------------------------
        # Print metrics
        # ----------------------------------------------------

        print(
            f"  Accuracy:          "
            f"{metrics['Accuracy']:.6f}"
        )

        print(
            f"  Balanced Accuracy: "
            f"{metrics['BalancedAccuracy']:.6f}"
        )

        print(
            f"  Precision:         "
            f"{metrics['Precision']:.6f}"
        )

        print(
            f"  Recall:            "
            f"{metrics['Recall']:.6f}"
        )

        print(
            f"  F1:                "
            f"{metrics['F1']:.6f}"
        )

        print(
            f"  ROC-AUC:           "
            f"{metrics['ROC_AUC']:.6f}"
        )

        print(
            f"  PR-AUC:            "
            f"{metrics['PR_AUC']:.6f}"
        )

        print()


    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )


    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    metric_names = [

        "Accuracy",

        "BalancedAccuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC",

        "PR_AUC"
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


# ===================================================================================================
# MAIN TRAINING / EVALUATION LOOP
# ===================================================================================================

all_fold_results = []

all_summary_results = []


for model_name in models:

    print()
    print()
    print("#" * 70)

    print(
        f"MODEL: {model_name}"
    )

    print("#" * 70)


    for method_name, method in eval_methods.items():

        splitter = method["splitter"]

        groups = method["groups"]


        # ----------------------------------------------------
        # Search for best parameters
        # ----------------------------------------------------

        best_params = search_best_parameters(

            model_name=model_name,

            method_name=method_name,

            splitter=splitter,

            groups=groups
        )


        # ----------------------------------------------------
        # Evaluate model
        # ----------------------------------------------------

        fold_results_df, summary = evaluate_model(

            model_name=model_name,

            method_name=method_name,

            splitter=splitter,

            groups=groups,

            best_params=best_params
        )


        # ----------------------------------------------------
        # Store fold results
        # ----------------------------------------------------

        all_fold_results.append(
            fold_results_df
        )


        # ----------------------------------------------------
        # Store summary
        # ----------------------------------------------------

        all_summary_results.append({

            "Model":
                model_name,

            "Method":
                method_name,

            **summary
        })


# ===================================================================================================
# COMBINE RESULTS
# ===================================================================================================

all_fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True
)

summary_df = pd.DataFrame(
    all_summary_results
)


# ===================================================================================================
# DISPLAY RESULTS
# ===================================================================================================

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


# ===================================================================================================
# SAVE EVALUATION RESULTS
# ===================================================================================================

all_fold_results_df.to_csv(

    EVALUATIONS_DIR +

    "AllModels_FoldResults.csv",

    index=False
)


summary_df.to_csv(

    EVALUATIONS_DIR +

    "AllModels_EvaluationSummary.csv",

    index=False
)


print()

print(
    "Evaluation results saved."
)


# ===================================================================================================
# TRAIN FINAL MODELS ON ALL DATA
# ===================================================================================================

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

            PARAMETERS_DIR +

            f"{model_name}_{method_name}.json"
        )


        model_path = (

            MODELS_DIR +

            f"{model_name}_{method_name}.joblib"
        )


        # ----------------------------------------------------
        # Load best parameters
        # ----------------------------------------------------

        with open(
            parameter_path,
            "r"
        ) as file:

            best_params = json.load(
                file
            )


        # ----------------------------------------------------
        # Create fresh model
        # ----------------------------------------------------

        final_model = create_model_with_params(

            model_name,

            best_params
        )


        # ----------------------------------------------------
        # Train on ALL data
        # ----------------------------------------------------

        final_model.fit(
            X,
            y
        )


        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        joblib.dump(
            final_model,
            model_path
        )


        print(
            f"{model_name} / {method_name}"
        )

        print(
            f"  Saved to: {model_path}"
        )


# ===================================================================================================
# PREFERRED CELL LINE + TIME COMBINATIONS
# ===================================================================================================

print()

print("=" * 70)

print(
    "PREFERRED SCOPOLETIN AND IMPERATORIN SAMPLES"
)

print("=" * 70)

print()

print(
    "Finding CellLine + Time combinations "
    "that appear among multiple Coumarins."
)


cellline_time_summary = (

    data.groupby(
        [
            "CellLine",
            "Time"
        ]
    )["Coumarin"]

    .nunique()

    .sort_values(
        ascending=False
    )

    .reset_index(
        name="NumberOfCoumarins"
    )
)


cellline_time_summary = (

    cellline_time_summary[

        cellline_time_summary[
            "NumberOfCoumarins"
        ] >= 3

    ]
)


print()

print(
    "CellLine + Time combinations ranked "
    "by number of Coumarins:"
)

print()

print(
    tabulate(
        cellline_time_summary,
        headers="keys",
        tablefmt="rounded_grid"
    )
)


# ===================================================================================================
# END
# ===================================================================================================

print()

print("=" * 70)

print(
    "FINISHED MULTI-MODEL CLASSIFICATION SCRIPT SUCCESSFULLY!!!"
)

print("=" * 70)

print()