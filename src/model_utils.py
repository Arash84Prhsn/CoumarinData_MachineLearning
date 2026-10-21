from pathlib import Path
import json
import re
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

RANDOM_STATE = 42
TARGET_COLUMN = "Viability"
GROUP_COLUMN = "Coumarin"
CELL_LINE_COLUMN = "CellLine"

DEFAULT_N_SPLITS = 5
DEFAULT_N_ITER = 100
DEFAULT_COVERAGE_THRESHOLDS = [25, 50, 75, 100]
DEFAULT_MIN_TRAINING_COUMARINS = 4

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

EXPERIMENTAL_FEATURES = ["LogDose", "Time"]
CATEGORICAL_FEATURES = [CELL_LINE_COLUMN]
DESCRIPTOR_FEATURES = EXPERIMENTAL_FEATURES + MOLECULAR_FEATURES


def normalize_identifiers(data):
    data = data.copy()

    for column in [GROUP_COLUMN, CELL_LINE_COLUMN]:
        if column in data.columns:
            data[column] = data[column].astype("string").str.strip().str.lower()

    return data


def get_morgan_features(data):
    return sorted(
        [column for column in data.columns if re.fullmatch(r"Morgan_\d+", column)],
        key=lambda column: int(column.split("_")[1]),
    )


def get_feature_sets(data):
    missing = [
        column
        for column in DESCRIPTOR_FEATURES + CATEGORICAL_FEATURES
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    morgan_features = get_morgan_features(data)

    if not morgan_features:
        raise ValueError("No Morgan fingerprint columns were found.")

    return {
        "Descriptor": DESCRIPTOR_FEATURES + CATEGORICAL_FEATURES,
        "Morgan": DESCRIPTOR_FEATURES + morgan_features + CATEGORICAL_FEATURES,
    }


def validate_features(data, features, target=TARGET_COLUMN):
    required = features + [target]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        missing_values = data[required].isna().sum()
        missing_values = missing_values[missing_values > 0]
        raise ValueError(f"Missing values detected:\n{missing_values}")


def build_preprocessor(features):
    categorical = [
        column for column in CATEGORICAL_FEATURES if column in features
    ]
    numeric = [column for column in features if column not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "R2": np.nan,
        }

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
    }


def split_training_external(data, external_coumarins):
    data = normalize_identifiers(data)

    external_coumarins = [
        compound.strip().lower() for compound in external_coumarins
    ]

    external_mask = data[GROUP_COLUMN].isin(external_coumarins)

    training_data = data.loc[~external_mask].copy()
    external_data = data.loc[external_mask].copy()

    if training_data.empty:
        raise ValueError("Training dataset is empty.")

    if external_data.empty:
        raise ValueError("External dataset is empty.")

    overlap = set(training_data[GROUP_COLUMN].unique()) & set(
        external_data[GROUP_COLUMN].unique()
    )

    if overlap:
        raise ValueError(f"External compounds leaked into training: {overlap}")

    return training_data, external_data


def run_cv_experiment(
    data,
    features,
    estimator,
    param_distributions,
    validation="KFold",
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
):
    validate_features(data, features)

    X = data[features]
    y = data[TARGET_COLUMN]

    if validation == "KFold":
        outer_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        inner_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        groups = None

    elif validation == "LOCO":
        outer_cv = LeaveOneGroupOut()
        inner_cv = LeaveOneGroupOut()
        groups = data[GROUP_COLUMN]

    else:
        raise ValueError("validation must be 'KFold' or 'LOCO'.")

    predictions = []
    fold_results = []

    splits = (
        outer_cv.split(X, y)
        if groups is None
        else outer_cv.split(X, y, groups)
    )

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if groups is not None:
            train_groups = groups.iloc[train_idx]
            held_out = groups.iloc[test_idx].iloc[0]

            print(
                f"{validation} fold {fold}: "
                f"held-out coumarin = {held_out}"
            )
        else:
            train_groups = None
            held_out = None

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="r2",
            cv=inner_cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )

        if train_groups is None:
            search.fit(X_train, y_train)
        else:
            search.fit(X_train, y_train, groups=train_groups)

        y_pred = search.predict(X_test)

        fold_predictions = pd.DataFrame(
            {
                "RowIndex": data.index[test_idx],
                "Actual": y_test.to_numpy(),
                "Predicted": y_pred,
                "Fold": fold,
                "Validation": validation,
                GROUP_COLUMN: data[GROUP_COLUMN].iloc[test_idx].to_numpy(),
            }
        )

        predictions.append(fold_predictions)

        fold_metrics = calculate_metrics(y_test, y_pred)
        fold_metrics.update(
            {
                "Fold": fold,
                "Validation": validation,
                "HeldOutCoumarin": held_out,
                # "BestParameters": str(search.best_params_),
            }
        )
        fold_results.append(fold_metrics)

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_results_df = pd.DataFrame(fold_results)

    summary = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )
    summary.update(
        {
            "Validation": validation,
            "FeatureCount": len(features),
        }
    )

    return {
        "Predictions": predictions_df,
        "FoldResults": fold_results_df,
        "Summary": pd.DataFrame([summary]),
    }


def fit_final_model(
    data,
    features,
    estimator,
    param_distributions,
    validation="KFold",
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
):
    validate_features(data, features)

    X = data[features]
    y = data[TARGET_COLUMN]

    if validation == "KFold":
        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        groups = None

    elif validation == "LOCO":
        cv = LeaveOneGroupOut()
        groups = data[GROUP_COLUMN]

    else:
        raise ValueError("validation must be 'KFold' or 'LOCO'.")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="r2",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    if groups is None:
        search.fit(X, y)
    else:
        search.fit(X, y, groups=groups)

    return search.best_estimator_, search.best_params_


def calculate_training_coverage(training_data):
    coverage = (
        training_data.groupby(CELL_LINE_COLUMN)
        .agg(
            TrainingObservations=(TARGET_COLUMN, "size"),
            TrainingCoumarins=(GROUP_COLUMN, "nunique"),
        )
        .reset_index()
    )

    return coverage


def evaluate_external_model(
    model,
    training_data,
    external_data,
    features,
    coverage_thresholds=DEFAULT_COVERAGE_THRESHOLDS,
    min_training_coumarins=DEFAULT_MIN_TRAINING_COUMARINS,
):
    validate_features(training_data, features)
    validate_features(external_data, features)

    external_predictions = external_data[
        [GROUP_COLUMN, CELL_LINE_COLUMN, TARGET_COLUMN]
    ].copy()

    external_predictions["Predicted"] = model.predict(
        external_data[features]
    )

    overall_metrics = calculate_metrics(
        external_predictions[TARGET_COLUMN],
        external_predictions["Predicted"],
    )

    overall_results = pd.DataFrame([overall_metrics])

    by_coumarin = []

    for coumarin, group in external_predictions.groupby(GROUP_COLUMN):
        metrics = calculate_metrics(
            group[TARGET_COLUMN],
            group["Predicted"],
        )
        metrics[GROUP_COLUMN] = coumarin
        by_coumarin.append(metrics)

    by_coumarin_results = pd.DataFrame(by_coumarin)

    coverage = calculate_training_coverage(training_data)

    external_predictions = external_predictions.merge(
        coverage,
        on=CELL_LINE_COLUMN,
        how="left",
    )

    external_predictions[
        ["TrainingObservations", "TrainingCoumarins"]
    ] = external_predictions[
        ["TrainingObservations", "TrainingCoumarins"]
    ].fillna(0)

    coverage_summary = []
    coverage_by_coumarin = []

    for threshold in coverage_thresholds:
        eligible = external_predictions[
            (external_predictions["TrainingObservations"] >= threshold)
            & (
                external_predictions["TrainingCoumarins"]
                >= min_training_coumarins
            )
        ]

        metrics = calculate_metrics(
            eligible[TARGET_COLUMN],
            eligible["Predicted"],
        )
        metrics["MinimumTrainingObservations"] = threshold
        metrics["MinimumTrainingCoumarins"] = min_training_coumarins
        coverage_summary.append(metrics)

        for coumarin, group in eligible.groupby(GROUP_COLUMN):
            compound_metrics = calculate_metrics(
                group[TARGET_COLUMN],
                group["Predicted"],
            )
            compound_metrics[GROUP_COLUMN] = coumarin
            compound_metrics["MinimumTrainingObservations"] = threshold
            compound_metrics["MinimumTrainingCoumarins"] = (
                min_training_coumarins
            )
            coverage_by_coumarin.append(compound_metrics)

    return {
        "Overall": overall_results,
        "ByCoumarin": by_coumarin_results,
        "CoverageSummary": pd.DataFrame(coverage_summary),
        "CoverageByCoumarin": pd.DataFrame(coverage_by_coumarin).sort_values(['Coumarin', 'MinimumTrainingObservations']),
        "ExternalPredictions": external_predictions,
    }


def save_model_results(
    results,
    output_directory,
    model_name,
    best_parameters=None,
    fitted_model=None,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for name, result in results.items():
        if isinstance(result, pd.DataFrame):
            result.to_csv(
                output_directory / f"{model_name}_{name}.csv",
                index=False,
            )

    if best_parameters is not None:
        with open(
            output_directory / f"{model_name}_BestParameters.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                best_parameters,
                file,
                indent=4,
                ensure_ascii=False,
                default=str,
            )

    if fitted_model is not None:
        joblib.dump(
            fitted_model,
            output_directory / f"{model_name}.joblib",
        )
        
def print_dataframe(df: pd.DataFrame):
    print(tabulate(df, headers='keys', tablefmt='rounded_grid'))
    
def build_scaled_preprocessor(features):
    categorical = [column for column in CATEGORICAL_FEATURES if column in features]
    numeric = [column for column in features if column not in categorical]

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
    )