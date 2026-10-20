import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import loguniform

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Data/Processed_Data.csv"
OUTPUT_DIR = "Results/MLP"

RANDOM_STATE = 42
OUTER_FOLDS = 5
N_ITER = 1

# Compounds reserved exclusively for external testing.
EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
]

MOLECULAR_FEATURES = [
    "MW",
    "LogP",
    "NumHDonors",
    "NumHAcceptors",
    "RotatableBonds",
    "TPSA",
    "RingCounts",
    "fCsp3",
]

# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

PARAM_DISTRIBUTIONS = {
    "regressor__mlp__hidden_layer_sizes": [
        (32,),
        (64,),
        (128,),
        (64, 32),
        (128, 64),
        (64, 32, 16),
    ],
    "regressor__mlp__activation": [
        "relu",
        "tanh",
    ],
    "regressor__mlp__alpha": loguniform(1e-5, 1e-1),
    "regressor__mlp__learning_rate_init": loguniform(
        1e-4,
        1e-2,
    ),
    "regressor__mlp__batch_size": [
        32,
        64,
        128,
        "auto",
    ],
}

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

data["Coumarin"] = (
    data["Coumarin"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.lower()
)

# Identify Morgan fingerprint columns.
fingerprint_columns = sorted(
    [
        column
        for column in data.columns
        if column.startswith("Morgan_")
    ],
    key=lambda column: int(column.split("_")[1]),
)

if not fingerprint_columns:
    raise ValueError(
        "No Morgan fingerprint columns were found. "
        "Check Data/Processed_Data.csv."
    )

required_columns = [
    "Coumarin",
    "CellLine",
    "Dose",
    "Time",
    "Viability",
    *MOLECULAR_FEATURES,
    *fingerprint_columns,
]

missing_columns = sorted(
    set(required_columns) - set(data.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

numeric_columns = [
    "Dose",
    "Time",
    "Viability",
    *MOLECULAR_FEATURES,
    *fingerprint_columns,
]

data[numeric_columns] = data[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

data = data.replace([np.inf, -np.inf], np.nan)

if data[required_columns].isna().any().any():
    raise ValueError(
        "Missing or non-finite values were found "
        "in required columns."
    )

if (data["Dose"] < 0).any() or (data["Time"] < 0).any():
    raise ValueError("Negative Dose or Time values were found.")

# ============================================================
# ENGINEER FEATURES
# ============================================================

data["LogDose"] = np.log1p(data["Dose"])

EXPERIMENTAL_FEATURES = [
    "LogDose",
    "Time",
]

CATEGORICAL_FEATURES = [
    "CellLine",
]

DESCRIPTOR_FEATURES = [
    *EXPERIMENTAL_FEATURES,
    *MOLECULAR_FEATURES,
]

MORGAN_FEATURES = [
    *DESCRIPTOR_FEATURES,
    *fingerprint_columns,
]

# ============================================================
# SEPARATE TRAINING AND EXTERNAL TEST DATA
# ============================================================

external_mask = data["Coumarin"].isin(EXTERNAL_COUMARINS)

train_data = (
    data.loc[~external_mask]
    .copy()
    .reset_index(drop=True)
)

external_data = (
    data.loc[external_mask]
    .copy()
    .reset_index(drop=True)
)

if train_data.empty or external_data.empty:
    raise ValueError(
        "The training or external test dataset is empty."
    )

missing_external = (
    set(EXTERNAL_COUMARINS)
    - set(external_data["Coumarin"].unique())
)

if missing_external:
    raise ValueError(
        "External compounds missing from the data: "
        f"{missing_external}"
    )

print("=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print(f"Total observations: {len(data)}")
print(f"Training observations: {len(train_data)}")
print(f"External observations: {len(external_data)}")

print("\nTraining coumarins:")
print(
    train_data["Coumarin"]
    .value_counts()
    .sort_index()
)

print("\nExternal coumarins:")
print(
    external_data["Coumarin"]
    .value_counts()
    .sort_index()
)

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ============================================================
# BUILD MLP PIPELINE
# ============================================================

def build_mlp_pipeline(numeric_features):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cellline",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    mlp_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "mlp",
                MLPRegressor(
                    solver="adam",
                    max_iter=2000,
                    tol=1e-4,
                    n_iter_no_change=30,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Standardize the target during fitting and transform
    # predictions back to the original viability scale.
    return TransformedTargetRegressor(
        regressor=mlp_pipeline,
        transformer=StandardScaler(),
    )


# ============================================================
# LOCO-BASED RANDOM SEARCH
# ============================================================

def select_hyperparameters(
    X,
    y,
    groups,
):
    """
    Tune the MLP using 100 randomized configurations.

    Every inner validation fold holds out one entire coumarin.
    MAE is calculated on the original viability scale.
    """

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError(
            "At least two coumarins are required "
            "for LOCO hyperparameter selection."
        )

    # The estimator is supplied through param_grid-independent
    # construction below because the selected feature set varies.
    # Its pipeline is already defined by the caller's X columns.
    raise NotImplementedError(
        "Use select_components_and_features below."
    )


def select_mlp_hyperparameters(
    X,
    y,
    groups,
    numeric_features,
):
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError(
            "At least two coumarins are required "
            "for LOCO hyperparameter selection."
        )

    estimator = build_mlp_pipeline(numeric_features)

    inner_cv = GroupKFold(
        n_splits=n_groups,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score="raise",
        return_train_score=False,
    )

    search.fit(
        X,
        y,
        groups=groups,
    )

    return search


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def evaluate_nested_cv(
    training_data,
    numeric_features,
    model_name,
    validation_strategy,
):
    X = training_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y = training_data["Viability"].to_numpy()
    groups = training_data["Coumarin"].to_numpy()

    # --------------------------------------------------------
    # Define outer validation strategy.
    # --------------------------------------------------------

    if validation_strategy == "LOCO":
        outer_cv = GroupKFold(
            n_splits=pd.Series(groups).nunique(),
        )

        outer_splits = outer_cv.split(
            X,
            y,
            groups=groups,
        )

    elif validation_strategy == "KFold":
        outer_cv = KFold(
            n_splits=OUTER_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        outer_splits = outer_cv.split(X, y)

    else:
        raise ValueError(
            f"Unknown validation strategy: "
            f"{validation_strategy}"
        )

    prediction_records = []
    fold_records = []

    for fold, (train_idx, valid_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        groups_train = groups[train_idx]

        # Hyperparameter selection uses LOCO in every case.
        search = select_mlp_hyperparameters(
            X=X_train,
            y=y_train,
            groups=groups_train,
            numeric_features=numeric_features,
        )

        predictions = np.asarray(
            search.predict(X_valid)
        ).ravel()

        fold_metrics = calculate_metrics(
            y_valid,
            predictions,
        )

        fold_record = {
            "Model": model_name,
            "Validation": validation_strategy,
            "Fold": fold,
            "BestInnerLOCOMAE": -search.best_score_,
            **fold_metrics,
        }

        if validation_strategy == "LOCO":
            held_out_coumarin = np.unique(
                groups[valid_idx]
            )

            if len(held_out_coumarin) != 1:
                raise RuntimeError(
                    "Expected exactly one held-out coumarin "
                    "in an outer LOCO fold."
                )

            fold_record["HeldOutCoumarin"] = (
                held_out_coumarin[0]
            )

        fold_records.append(fold_record)

        for idx, actual, predicted in zip(
            valid_idx,
            y_valid,
            predictions,
        ):
            prediction_records.append(
                {
                    "Model": model_name,
                    "Validation": validation_strategy,
                    "Fold": fold,
                    "Coumarin": groups[idx],
                    "CellLine": training_data.iloc[idx]["CellLine"],
                    "Actual": actual,
                    "Predicted": predicted,
                    "Error": predicted - actual,
                }
            )

        held_out_label = ""

        if validation_strategy == "LOCO":
            held_out_label = (
                f" | Held out: {held_out_coumarin[0]}"
            )

        print(
            f"{model_name} | {validation_strategy}"
            f" | Fold {fold}{held_out_label}"
            f" | Best inner LOCO MAE: "
            f"{-search.best_score_:.4f}"
            f" | MAE: {fold_metrics['MAE']:.4f}"
            f" | RMSE: {fold_metrics['RMSE']:.4f}"
            f" | R2: {fold_metrics['R2']:.4f}"
        )

    predictions_df = pd.DataFrame(prediction_records)
    folds_df = pd.DataFrame(fold_records)

    overall_metrics = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )

    overall_record = {
        "Model": model_name,
        "Validation": validation_strategy,
        **overall_metrics,
    }

    return (
        predictions_df,
        folds_df,
        overall_record,
    )


# ============================================================
# EVALUATE BOTH MLP REPRESENTATIONS
# ============================================================

model_features = {
    "Descriptor_MLP": DESCRIPTOR_FEATURES,
    "Morgan_MLP": MORGAN_FEATURES,
}

all_predictions = []
all_fold_results = []
all_summary_results = []

for model_name, numeric_features in model_features.items():
    for validation_strategy in ["KFold", "LOCO"]:
        print("\n" + "=" * 80)
        print(
            f"{model_name} | {validation_strategy}"
        )
        print("=" * 80)

        (
            predictions_df,
            folds_df,
            summary,
        ) = evaluate_nested_cv(
            training_data=train_data,
            numeric_features=numeric_features,
            model_name=model_name,
            validation_strategy=validation_strategy,
        )

        all_predictions.append(predictions_df)
        all_fold_results.append(folds_df)
        all_summary_results.append(summary)

# ============================================================
# SAVE CROSS-VALIDATION RESULTS
# ============================================================

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True,
)

summary_df = pd.DataFrame(all_summary_results)

predictions_df.to_csv(
    f"{OUTPUT_DIR}/CV_Predictions.csv",
    index=False,
)

fold_results_df.to_csv(
    f"{OUTPUT_DIR}/CV_Fold_Results.csv",
    index=False,
)

summary_df.to_csv(
    f"{OUTPUT_DIR}/CV_Summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# ============================================================
# FINAL FIT AND EXTERNAL EVALUATION
# ============================================================

external_predictions = []

for model_name, numeric_features in model_features.items():
    print("\n" + "=" * 80)
    print(f"FINAL MODEL: {model_name}")
    print("=" * 80)

    X_train = train_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y_train = train_data["Viability"].to_numpy()
    groups_train = train_data["Coumarin"].to_numpy()

    X_external = external_data[
        numeric_features + CATEGORICAL_FEATURES
    ].copy()

    y_external = external_data["Viability"].to_numpy()

    # Tune on training coumarins only.
    final_search = select_mlp_hyperparameters(
        X=X_train,
        y=y_train,
        groups=groups_train,
        numeric_features=numeric_features,
    )

    final_model = final_search.best_estimator_

    print(
        f"Best inner LOCO MAE: "
        f"{-final_search.best_score_:.4f}"
    )

    print("\nBest hyperparameters:")

    for parameter, value in sorted(
        final_search.best_params_.items()
    ):
        print(f"{parameter}: {value}")

    # Save best hyperparameters.
    best_params = {
        key: (
            list(value)
            if isinstance(value, tuple)
            else value
        )
        for key, value in final_search.best_params_.items()
    }

    with open(
        f"{OUTPUT_DIR}/{model_name}_BestParameters.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_params,
            file,
            indent=4,
        )

    # Predict the external observations.
    y_pred = np.asarray(
        final_model.predict(X_external)
    ).ravel()

    external_metrics = calculate_metrics(
        y_external,
        y_pred,
    )

    print("\nOverall external metrics:")

    for metric, value in external_metrics.items():
        if metric == "N":
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    # --------------------------------------------------------
    # Save individual external predictions.
    # --------------------------------------------------------

    for i, (_, row) in enumerate(
        external_data.iterrows()
    ):
        external_predictions.append(
            {
                "Model": model_name,
                "Coumarin": row["Coumarin"],
                "CellLine": row["CellLine"],
                "Dose": row["Dose"],
                "Time": row["Time"],
                "Actual": y_external[i],
                "Predicted": y_pred[i],
                "Error": y_pred[i] - y_external[i],
            }
        )

    # --------------------------------------------------------
    # External performance by coumarin.
    # --------------------------------------------------------

    compound_results = []

    for coumarin in EXTERNAL_COUMARINS:
        mask = (
            external_data["Coumarin"].to_numpy()
            == coumarin
        )

        if not mask.any():
            continue

        compound_metrics = calculate_metrics(
            y_external[mask],
            y_pred[mask],
        )

        compound_results.append(
            {
                "Model": model_name,
                "Coumarin": coumarin,
                **compound_metrics,
            }
        )

    compound_results_df = pd.DataFrame(
        compound_results
    )

    print("\nExternal performance by coumarin:")

    print(
        compound_results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    compound_results_df.to_csv(
        f"{OUTPUT_DIR}/{model_name}_External_By_Coumarin.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save final fitted model.
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        f"{OUTPUT_DIR}/{model_name}_Final.joblib",
    )

# ============================================================
# SAVE ALL EXTERNAL PREDICTIONS
# ============================================================

external_predictions_df = pd.DataFrame(
    external_predictions
)

external_predictions_df.to_csv(
    f"{OUTPUT_DIR}/External_Predictions.csv",
    index=False,
)

print("\n" + "=" * 80)
print("MLP EVALUATION COMPLETE")
print("=" * 80)

print(f"Results directory: {OUTPUT_DIR}/")

print("\nSaved files:")
print("- CV_Predictions.csv")
print("- CV_Fold_Results.csv")
print("- CV_Summary.csv")
print("- External_Predictions.csv")
print("- Descriptor_MLP_External_By_Coumarin.csv")
print("- Morgan_MLP_External_By_Coumarin.csv")
print("- Descriptor_MLP_BestParameters.json")
print("- Morgan_MLP_BestParameters.json")
print("- Descriptor_MLP_Final.joblib")
print("- Morgan_MLP_Final.joblib")