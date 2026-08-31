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
RESULTS_DIR = "Results/Regression/XGboost/"
MODELS_DIR = RESULTS_DIR + "Models/"
EVALUATIONS_DIR = RESULTS_DIR + "Evaluations/"
PARAMETERS_DIR = RESULTS_DIR + "BestParameters/"
N_ITER = 3000 # Number of iterations for the random search

# Make the required directories if they do not exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PARAMETERS_DIR, exist_ok=True)

print("=" * 70)
print("STARTING XGBOOST REGRESSION SCRIPT")
print("=" * 70)

print()
print("Head of the raw data: (Shuffled to see more variance in the data)\n")
print(tabulate(data.sample(frac=1).head(), headers='keys', tablefmt="rounded_grid"))
print()

# Set X and y
print()
print("=" * 70)
print("DEFINING X AND y")
print("=" * 70)
print()
X = data.drop(columns=["Viability", "Author", "Coumarin"], inplace=False)
y = data['Viability']

print("X features:")
print(tabulate(pd.DataFrame(X.columns), tablefmt="rounded_grid"), end="\n\n")
print("The y vector has been set to the `Viability` feature")

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
print()
print("=" * 70)
print("Defining Grouping Strategies")
print("=" * 70)
print()

print("1. Grouping using the Author, Coumarin, CellLinem and Time Features")
print("Number of uniquely identified groups:", data["GroupID"].nunique())
print()

print("2. Grouping using the Coumarin feature")
print("Number Groups using only Coumarins:", data["Coumarin"].nunique())
print()

print("3. The final strategy will be the usual Kfold using 5 folds")

# Prepare the preprocessor
preprocessor = ColumnTransformer(
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

# Initialize the base model object
model = XGBRegressor(
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Define pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Define the search grid of the model for fine tuning the hyper parameters (Using randomSearch)
param_distributions = {
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

# Next up we'll define some helpers so that reproducubility under different conditions is simplified

#===================================================================================================
# Helper function to get the results of evaluation
def calculate_metrics(y_true, y_pred):
    """
    Returns a Dict containing the following metrics as keys:
    - "MAE"
    - "RMSE"
    - "MedAE"
    - "R2"

    :param y_true: The true y vector.
    :param y_pred: The y that has been predicted by a model.
    """
    return {
        "MAE" : mean_absolute_error(y_true, y_pred),
        "RMSE" : np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE" : median_absolute_error(y_true, y_pred),
        "R2" : r2_score(y_true, y_pred)
    };

# A helper function to return a new pipeline if needed
def create_model():

    return Pipeline([
        (
            "preprocessor",
            ColumnTransformer(
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
        ),
        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                n_jobs=-1,
                random_state=RANDOM_STATE
            )
        )
    ]);

# Helper function that will return a pipeline that has been fitted to the given params
def create_model_with_params(best_params):

    model_pipeline = create_model()

    model_pipeline.set_params(
        **best_params
    )

    return model_pipeline;

# A Function that will allow us to perform the search under different conditions if needed
def search_best_parameters(method_name, splitter, groups):
    """
    Performs RandomizedSearchCV if the best-parameter JSON
    does not already exist.

    If the JSON exists, the search is skipped and the
    previously determined parameters are loaded instead.
    """

    parameter_path = (
        PARAMETERS_DIR +
        f"XGBoost_{method_name}.json"
    )

    print()
    print("=" * 70)
    print(f"Starting hyperparameter search: {method_name}")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Check whether the search has already been performed
    # --------------------------------------------------------

    if os.path.exists(parameter_path):

        print(f"[{method_name}] Best parameters already exist.")
        print("Skipping hyperparameter search.")

        with open(
            parameter_path,
            "r"
        ) as file:

            best_params = json.load(file)

        return best_params


    # --------------------------------------------------------
    # Perform the search if the json does not exist
    # --------------------------------------------------------

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=N_ITER,
        scoring="r2",
        cv=splitter,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=False,
        verbose=1
    )

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
    # Retrieve best parameters post searching
    # --------------------------------------------------------

    best_params = search.best_params_

    print()
    print(
        f"Best CV R2 for {method_name}: "
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
        f"Best parameters saved to: "
        f"{parameter_path}"
    )

    return best_params;

# A function that will allow us to find metrics other than the R2 used in the search
def evaluate_model(
    method_name,
    splitter,
    groups,
    best_params
):
    """
    Evaluates a model with fixed hyperparameters using the
    same CV strategy that was used during hyperparameter
    searching.

    A fresh model is fitted for every fold.

    Returns:
        - fold-level results
        - mean metrics
        - standard deviations
    """

    print()
    print("=" * 70)
    print(
        f"Evaluating best model: using {method_name} splitting method"
    )
    print("=" * 70)
    print()

    fold_results = []

    # --------------------------------------------------------
    # Generate the same CV splits used during the search
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

    for fold_number, (train_idx, test_idx) in enumerate(
        split_iterator,
        start=1
    ):

        print(
            f"Evaluating fold {fold_number}..."
        )

        # Find the left out coumarin in case that we are doing the coumarin method.
        left_out_coumarin = None
        if method_name == "Coumarin" :
            left_out_coumarin = str(data.iloc[test_idx]["Coumarin"].iloc[0])
            print("Left Out Coumarin is:", left_out_coumarin)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fresh model for this fold
        fold_model = create_model_with_params(
            best_params
        )

        # Fit ONLY on the training portion
        fold_model.fit(
            X_train,
            y_train
        )

        # Predict the held-out portion
        y_pred = fold_model.predict(
            X_test
        )

        # Calculate all metrics
        metrics = calculate_metrics(
            y_test,
            y_pred
        )

        fold_results.append({
            "Method": method_name,
            "Fold": fold_number,
            "Left Out Coumarin" : left_out_coumarin,
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

    # --------------------------------------------------------
    # Convert fold results to DataFrame
    # --------------------------------------------------------

    fold_results_df = pd.DataFrame(
        fold_results
    )

    # --------------------------------------------------------
    # Calculate mean and standard deviation
    # --------------------------------------------------------

    metrics = [
        "MAE",
        "RMSE",
        "MedAE",
        "R2"
    ]

    summary = {}

    for metric in metrics:

        summary[
            f"{metric}_Mean"
        ] = fold_results_df[metric].mean()

        summary[
            f"{metric}_Std"
        ] = fold_results_df[metric].std()

    return fold_results_df, summary;
#===================================================================================================

# We are done with the helper functions now we can start the process of searching and evaluating

all_fold_results = []
all_summary_results = []

for method_name, method in eval_methods.items():
    
    # Set the splitter and the groups to be used in the search and evaluation
    splitter = method["splitter"]
    groups = method["groups"]

    # Get the best_params from the model
    best_params = search_best_parameters(
        method_name=method_name,
        splitter=splitter,
        groups=groups
    )

    # Get the results of the evaluation
    fold_results_df, summary = evaluate_model(
        method_name=method_name,
        splitter=splitter,
        groups=groups,
        best_params=best_params
    )

    # Store fold level results
    all_fold_results.append(fold_results_df)

    # Store our summary
    all_summary_results.append({
        "Method" : method_name,
        **summary # unpack the summary dict
    });


# Lets combine the results to one df
all_fold_results_df = pd.concat(all_fold_results, ignore_index=True)
summary_df = pd.DataFrame(all_summary_results)

# Log the results in the terminal
print()
print("=" * 70)
print("FOLD-LEVEL EVALUATION RESULTS")
print("=" * 70)
print()

print(tabulate(all_fold_results_df, headers="keys", tablefmt="rounded_grid", showindex=False, floatfmt=".6f"))

print()
print("=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)
print()

print(tabulate(summary_df, headers="keys", tablefmt="rounded_grid", showindex=False, floatfmt=".6f"))

# Now we can finally save the results as a csv file
all_fold_results_df.to_csv(
    EVALUATIONS_DIR +
    "XGBoost_FoldResults.csv",
    index=False
)

summary_df.to_csv(
    EVALUATIONS_DIR +
    "XGBoost_EvaluationSummary.csv",
    index=False
)

print()
print("Evaluation results saved.")

# Now that we are done with searching and evaluating, We can finally train the models on the whole dataset
print()
print("=" * 70)
print("TRAINING FINAL MODELS")
print("=" * 70)
print()


for method_name, method in eval_methods.items():

    parameter_path = (
        PARAMETERS_DIR +
        f"XGBoost_{method_name}.json"
    )

    model_path = (
        MODELS_DIR +
        f"XGBoost_{method_name}.joblib"
    )

    # --------------------------------------------------------
    # Load best parameters
    # --------------------------------------------------------

    with open(
        parameter_path,
        "r"
    ) as file:

        best_params = json.load(file)

    # --------------------------------------------------------
    # Create fresh model
    # --------------------------------------------------------

    final_model = create_model_with_params(
        best_params
    )

    # --------------------------------------------------------
    # Train on ALL available data
    # --------------------------------------------------------

    final_model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        model_path
    )

    print(f"{method_name} final model saved to:")
    print(f"  {model_path}")

print()
print("=" * 70)
print("PREFERED SCOPOLETIN AND IMPERATORIN SAMPLES")
print("=" * 70)
print()

print("In order to find the prefered samples, we wanted to see which CellLine and Timepoint combinations appear among most Coumarins.")

cellline_time_summary = (
    data.groupby(["CellLine", "Time"])["Coumarin"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="NumberOfCoumarins")
)

# Consider combinations that appear among at least 4 coumarins:
cellline_time_summary = cellline_time_summary[cellline_time_summary["NumberOfCoumarins"] >= 3]

print("\nCellLine + Time combinations ranked by number of Coumarins:")
print(tabulate(cellline_time_summary, headers='keys', tablefmt='rounded_grid'))

# We can now move onto validating and testing our model on new coumarins and samples not in the dataset
print()
print("=" * 70)
print("Loading Validation Samples")
print("=" * 70)
print()


print()
print("=" * 70)
print("TESTING THE MODEL ON FARNESIFEROL_C SAMPLES")
print("=" * 70)
print()

farenesiferolC_samples = [{
    "Cancer" : [],
    "CellLine" : [],
    "Dose" : [],
    "Time" : [],
    "Viability" : []
    },
]

farenesiferolC_structural_features = {
    "MW" : 382.5, # From PubChem
    "LogP" : 5.5, # From PubChem
    "NumHDonors" : 0, # From PubChem
    "NumHAcceptors" : 4, # From PubChem
    "RotatableBonds" : 6, # From PubChem
    "TPSA" : 44.8, # From PubChem
    "RingCounts" : 2, # From PubChem Figure
    "fCsp3" : 0.54 # From SwissADME
}


print()
print("=" * 70)
print("TESTING THE MODEL ON SCOPOLETIN SAMPLES")
print("=" * 70)
print()

scopoletin_samples = [{
    "Cancer" : [],
    "CellLine" : [],
    "Dose" : [],
    "Time" : [],
    "Viability" : []
    },
]

scopoletin_structural_features = {
    "MW" : 192.17, # From PubChem
    "LogP" : 1.5, # From PubChem
    "NumHDonors" : 1, # From PubChem
    "NumHAcceptors" : 4, # From PubChem
    "RotatableBonds" : 1, # From PubChem
    "TPSA" : 55.8, # From PubChem
    "RingCounts" : 2, # From PubChem Figure
    "fCsp3" : 0.10 # From SwissADME
}

print()
print("=" * 70)
print("TESTING THE MODEL ON IMPERATORIN SAMPLES")
print("=" * 70)
print()

imperatorin_samples = [{
    "Cancer" : [],
    "CellLine" : [],
    "Dose" : [],
    "Time" : [],
    "Viability" : []
    },
]

imperatorin_structural_features = {
    "MW" : 270.28, # From PubChem
    "LogP" : 3.4, # From PubChem
    "NumHDonors" : 0, # From PubChem
    "NumHAcceptors" : 4, # From PubChem
    "RotatableBonds" : 3, # From PubChem
    "TPSA" : 48.7, # From PubChem
    "RingCounts" : 3, # From PubChem Figure
    "fCsp3" : 0.19 # From SwissADME
}


print()
print("=" * 70)
print("FINISHED SCRIPT SUCCESSFULLY!!!")
print("=" * 70)
print()