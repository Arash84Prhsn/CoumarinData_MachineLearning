import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "Data/Processed_Data.csv"
OUTPUT_DIR = "Results/DecisionTree"

RANDOM_STATE = 42
OUTER_FOLDS = 5
N_ITER = 100

# Compounds reserved exclusively for external testing.
EXTERNAL_COUMARINS = [
    "scopoletin",
    "imperatorin",
    "farnesiferola",
    "farnesiferolb",
    "farnesiferolc",
]

# External evaluation thresholds.
COVERAGE_THRESHOLDS = [25, 50, 75, 100]

# Cell lines must appear in at least this many distinct
# training coumarins.
MIN_TRAINING_COUMARINS = 4

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
    "tree__criterion": [
        "squared_error",
        "poisson",
        "absolute_error",
    ],
    "tree__max_depth": [
        None,
        3,
        5,
        8,
        10,
        15,
        20,
    ],
    "tree__min_samples_split": [
        2,
        5,
        10,
        15,
        20,
        30,
    ],
    "tree__min_samples_leaf": [
        1,
        2,
        4,
        6,
        10,
        15,
        20,
    ],
    "tree__max_features": [
        None,
        "sqrt",
        "log2",
        0.25,
        0.5,
        0.75,
    ],
    "tree__ccp_alpha": [
        0.0,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.02,
    ],
}

# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

data = pd.read_csv(DATA_PATH)

# Normalize categorical values.
data["Coumarin"] = (
    data["Coumarin"]
    .astype(str)
    .str.replace(r"\s+", "", regex=True)
    .str.lower()
)

data["CellLine"] = (
    data["CellLine"]
    .astype(str)
    .str.strip()
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

# Validate required columns.
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

# Validate numerical data.
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
    raise ValueError(
        "Negative Dose or Time values were found."
    )

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
# TRAINING CELL-LINE COVERAGE
# ============================================================

# All coverage statistics are calculated exclusively from
# the training data.

training_cellline_coverage = (
    train_data
    .groupby("CellLine")
    .agg(
        TrainingCoumarins=("Coumarin", "nunique"),
        TrainingSamples=("Coumarin", "size"),
    )
    .reset_index()
)

training_cellline_coverage.to_csv(
    f"{OUTPUT_DIR}/Training_CellLine_Coverage.csv",
    index=False,
)

# Attach training-derived coverage statistics to external data.
# This does not change the training dataset or model fitting.
external_data = external_data.merge(
    training_cellline_coverage,
    on="CellLine",
    how="left",
    validate="many_to_one",
)

external_data["TrainingCoumarins"] = (
    external_data["TrainingCoumarins"]
    .fillna(0)
    .astype(int)
)

external_data["TrainingSamples"] = (
    external_data["TrainingSamples"]
    .fillna(0)
    .astype(int)
)

external_data["MeetsCoumarinCoverage"] = (
    external_data["TrainingCoumarins"]
    >= MIN_TRAINING_COUMARINS
)

print("\n" + "=" * 80)
print("EXTERNAL CELL-LINE COVERAGE")
print("=" * 80)

print(
    external_data[
        [
            "Coumarin",
            "CellLine",
            "TrainingCoumarins",
            "TrainingSamples",
            "MeetsCoumarinCoverage",
        ]
    ]
    .drop_duplicates()
    .sort_values(["Coumarin", "CellLine"])
    .to_string(index=False)
)

print("\nEligible external observations:")

for threshold in COVERAGE_THRESHOLDS:
    mask = (
        external_data["MeetsCoumarinCoverage"]
        & (
            external_data["TrainingSamples"]
            >= threshold
        )
    )

    print(
        f"Threshold >= {threshold:3d}: "
        f"{mask.sum():4d} / {len(external_data)} observations"
    )

# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MedAE": median_absolute_error(y_true, y_pred),
    }

    if len(y_true) >= 2 and np.var(y_true) > 0:
        metrics["R2"] = r2_score(y_true, y_pred)
    else:
        metrics["R2"] = np.nan

    return metrics


# ============================================================
# BUILD DECISION TREE PIPELINE
# ============================================================

def build_tree_pipeline(numeric_features):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
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

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "tree",
                DecisionTreeRegressor(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# LOCO-BASED RANDOM SEARCH
# ============================================================

def select_hyperparameters(
    X,
    y,
    groups,
    numeric_features,
):
    """
    Select Decision Tree hyperparameters using
    100 randomized candidates and LOCO cross-validation.

    The optimization criterion is negative MAE.
    """

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))

    if n_groups < 2:
        raise ValueError(
            "At least two coumarins are required "
            "for LOCO hyperparameter selection."
        )

    pipeline = build_tree_pipeline(
        numeric_features
    )

    inner_cv = GroupKFold(
        n_splits=n_groups,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
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

        # Hyperparameter search uses LOCO even when the
        # outer evaluation strategy is ordinary KFold.
        search = select_hyperparameters(
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
                    "CellLine": (
                        training_data.iloc[idx]["CellLine"]
                    ),
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

    predictions_df = pd.DataFrame(
        prediction_records
    )

    folds_df = pd.DataFrame(
        fold_records
    )

    overall_metrics = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )

    overall_record = {
        "Model": model_name,
        "Validation": validation_strategy,
        **overall_metrics,
    }

    return predictions_df, folds_df, overall_record


# ============================================================
# EVALUATE BOTH FEATURE REPRESENTATIONS
# ============================================================

model_features = {
    "Descriptor_DecisionTree": DESCRIPTOR_FEATURES,
    "Morgan_DecisionTree": MORGAN_FEATURES,
}

all_predictions = []
all_fold_results = []
all_summary_results = []

for model_name, numeric_features in model_features.items():
    for validation_strategy in ["KFold", "LOCO"]:
        print("\n" + "=" * 80)
        print(f"{model_name} | {validation_strategy}")
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

cv_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

cv_fold_results_df = pd.concat(
    all_fold_results,
    ignore_index=True,
)

cv_summary_df = pd.DataFrame(
    all_summary_results
)

cv_predictions_df.to_csv(
    f"{OUTPUT_DIR}/CV_Predictions.csv",
    index=False,
)

cv_fold_results_df.to_csv(
    f"{OUTPUT_DIR}/CV_Fold_Results.csv",
    index=False,
)

cv_summary_df.to_csv(
    f"{OUTPUT_DIR}/CV_Summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY")
print("=" * 80)

print(
    cv_summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

# ============================================================
# FINAL FIT AND EXTERNAL EVALUATION
# ============================================================

external_predictions = []

coverage_summary_records = []
coverage_compound_records = []

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

    # Select hyperparameters using training coumarins only.
    final_search = select_hyperparameters(
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

    # Save best parameters, converting NumPy scalars if needed.
    best_params = {
        key: (
            value.item()
            if isinstance(value, np.generic)
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

    # Save the fitted model.
    joblib.dump(
        final_model,
        f"{OUTPUT_DIR}/{model_name}_Final.joblib",
    )

    # Predict all external observations.
    # Coverage filtering does not affect predictions or fitting.
    y_pred = np.asarray(
        final_model.predict(X_external)
    ).ravel()

    # --------------------------------------------------------
    # FULL EXTERNAL TEST METRICS
    # --------------------------------------------------------

    full_metrics = calculate_metrics(
        y_external,
        y_pred,
    )

    print("\nFull external test metrics:")

    for metric, value in full_metrics.items():
        if metric == "N":
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    # Save all external predictions with training coverage.
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
                "TrainingCoumarins": (
                    row["TrainingCoumarins"]
                ),
                "TrainingSamples": row["TrainingSamples"],
                "MeetsCoumarinCoverage": (
                    row["MeetsCoumarinCoverage"]
                ),
            }
        )

    # --------------------------------------------------------
    # FULL EXTERNAL RESULTS BY COUMARIN
    # --------------------------------------------------------

    full_compound_records = []

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

        full_compound_records.append(
            {
                "Model": model_name,
                "Coumarin": coumarin,
                **compound_metrics,
            }
        )

    full_compound_df = pd.DataFrame(
        full_compound_records
    )

    print("\nFull external performance by coumarin:")

    print(
        full_compound_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    full_compound_df.to_csv(
        f"{OUTPUT_DIR}/{model_name}_External_By_Coumarin.csv",
        index=False,
    )

    # --------------------------------------------------------
    # COVERAGE-FILTERED EXTERNAL EVALUATION
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"COVERAGE-FILTERED RESULTS: {model_name}")
    print("=" * 80)

    for threshold in COVERAGE_THRESHOLDS:

        # This filter affects external evaluation only.
        eligible_mask = (
            external_data["MeetsCoumarinCoverage"].to_numpy()
            & (
                external_data["TrainingSamples"].to_numpy()
                >= threshold
            )
        )

        subset_data = external_data.loc[
            eligible_mask
        ].copy()

        y_subset = y_external[eligible_mask]
        pred_subset = y_pred[eligible_mask]

        if len(subset_data) > 0:
            subset_metrics = calculate_metrics(
                y_subset,
                pred_subset,
            )

            n_celllines = subset_data["CellLine"].nunique()
            n_coumarins = subset_data["Coumarin"].nunique()

            print(
                f"\nThreshold >= {threshold}"
                f" | N: {len(subset_data)}"
                f" | Cell lines: {n_celllines}"
                f" | Coumarins: {n_coumarins}"
                f" | MAE: {subset_metrics['MAE']:.4f}"
                f" | RMSE: {subset_metrics['RMSE']:.4f}"
                f" | MedAE: {subset_metrics['MedAE']:.4f}"
                f" | R2: {subset_metrics['R2']:.4f}"
            )

            coverage_summary_records.append(
                {
                    "Model": model_name,
                    "Threshold": threshold,
                    "MinTrainingCoumarins": (
                        MIN_TRAINING_COUMARINS
                    ),
                    **subset_metrics,
                    "CellLines": n_celllines,
                    "Coumarins": n_coumarins,
                }
            )

        else:
            print(
                f"\nThreshold >= {threshold}"
                " | No eligible external observations."
            )

            coverage_summary_records.append(
                {
                    "Model": model_name,
                    "Threshold": threshold,
                    "MinTrainingCoumarins": (
                        MIN_TRAINING_COUMARINS
                    ),
                    "N": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MedAE": np.nan,
                    "R2": np.nan,
                    "CellLines": 0,
                    "Coumarins": 0,
                }
            )

        # ----------------------------------------------------
        # Report every coumarin at every threshold.
        # Even coumarins with zero eligible observations
        # receive a row with N=0 and undefined metrics.
        # ----------------------------------------------------

        print(f"\nResults by coumarin | Threshold >= {threshold}")

        for coumarin in EXTERNAL_COUMARINS:
            compound_mask = (
                subset_data["Coumarin"].to_numpy()
                == coumarin
            )

            if compound_mask.any():
                compound_actual = y_subset[compound_mask]
                compound_pred = pred_subset[compound_mask]

                compound_metrics = calculate_metrics(
                    compound_actual,
                    compound_pred,
                )

                record = {
                    "Model": model_name,
                    "Threshold": threshold,
                    "Coumarin": coumarin,
                    **compound_metrics,
                }

                print(
                    f"  {coumarin:20s}"
                    f" | N: {compound_metrics['N']:4d}"
                    f" | MAE: {compound_metrics['MAE']:.4f}"
                    f" | RMSE: {compound_metrics['RMSE']:.4f}"
                    f" | MedAE: {compound_metrics['MedAE']:.4f}"
                    f" | R2: {compound_metrics['R2']:.4f}"
                )

            else:
                record = {
                    "Model": model_name,
                    "Threshold": threshold,
                    "Coumarin": coumarin,
                    "N": 0,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "MedAE": np.nan,
                    "R2": np.nan,
                }

                print(
                    f"  {coumarin:20s}"
                    " | N:    0 | No eligible observations"
                )

            coverage_compound_records.append(record)

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

# ============================================================
# SAVE COVERAGE EVALUATION RESULTS
# ============================================================

coverage_summary_df = pd.DataFrame(
    coverage_summary_records
)

coverage_summary_df.to_csv(
    f"{OUTPUT_DIR}/External_Coverage_Subset_Summary.csv",
    index=False,
)

coverage_compound_df = pd.DataFrame(
    coverage_compound_records
)

coverage_compound_df.to_csv(
    f"{OUTPUT_DIR}/External_Coverage_By_Coumarin.csv",
    index=False,
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("DECISION TREE EVALUATION COMPLETE")
print("=" * 80)

print(f"Results directory: {OUTPUT_DIR}/")

print("\nSaved files:")
print("- CV_Predictions.csv")
print("- CV_Fold_Results.csv")
print("- CV_Summary.csv")
print("- External_Predictions.csv")
print("- Training_CellLine_Coverage.csv")
print("- External_Coverage_Subset_Summary.csv")
print("- External_Coverage_By_Coumarin.csv")
print("- Descriptor_DecisionTree_External_By_Coumarin.csv")
print("- Morgan_DecisionTree_External_By_Coumarin.csv")
print("- Descriptor_DecisionTree_BestParameters.json")
print("- Morgan_DecisionTree_BestParameters.json")
print("- Descriptor_DecisionTree_Final.joblib")
print("- Morgan_DecisionTree_Final.joblib")