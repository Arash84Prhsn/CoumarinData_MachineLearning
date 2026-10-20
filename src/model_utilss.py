from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "Viability"
GROUP_COLUMN = "Coumarin"
CELL_LINE_COLUMN = "CellLine"

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

EXPERIMENTAL_FEATURES = [
    "LogDose",
    "Time",
]

CATEGORICAL_FEATURES = [
    "CellLine",
]

DESCRIPTOR_FEATURES = (
    EXPERIMENTAL_FEATURES
    + MOLECULAR_FEATURES
)

COVERAGE_THRESHOLDS = [
    25,
    50,
    75,
    100,
]

MIN_TRAINING_COUMARINS = 4


# ============================================================
# FEATURE UTILITIES
# ============================================================

def get_morgan_features(data):
    """
    Return Morgan fingerprint column names in numerical order.
    """

    features = [
        column
        for column in data.columns
        if column.startswith("Morgan_")
    ]

    def fingerprint_index(column):
        return int(column.rsplit("_", 1)[1])

    return sorted(features, key=fingerprint_index)


def get_feature_sets(data):
    """
    Return the descriptor-only and Morgan feature sets.

    Raises an error if any expected descriptor is missing.
    """

    missing = [
        feature
        for feature in DESCRIPTOR_FEATURES
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required descriptor features: {missing}"
        )

    morgan_features = get_morgan_features(data)

    if not morgan_features:
        raise ValueError(
            "No Morgan fingerprint columns were found."
        )

    return {
        "Descriptor": DESCRIPTOR_FEATURES.copy(),
        "Morgan": (
            DESCRIPTOR_FEATURES
            + morgan_features
        ),
    }


def validate_features(data, features, target=TARGET_COLUMN):
    """
    Check that the requested features and target exist
    and contain no missing values.
    """

    missing_columns = [
        column
        for column in features + [target]
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if data[features + [target]].isna().any().any():
        raise ValueError(
            "Missing values found in the selected features or target."
        )

    return True


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics consistently across models.

    Returns MAE, RMSE, MedAE, and R2.
    R2 is NaN when it cannot be meaningfully calculated.
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MedAE": np.nan,
            "R2": np.nan,
        }

    if not (
        np.isfinite(y_true).all()
        and np.isfinite(y_pred).all()
    ):
        raise ValueError(
            "Targets and predictions must contain only finite values."
        )

    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) >= 2
        and np.var(y_true) > 0
        else np.nan
    )

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),
        "MedAE": median_absolute_error(
            y_true,
            y_pred,
        ),
        "R2": r2,
    }


# ============================================================
# DATA PREPARATION
# ============================================================

def normalize_identifiers(
    data,
    coumarin_column=GROUP_COLUMN,
    cell_line_column=CELL_LINE_COLUMN,
):
    """
    Normalize coumarin and cell-line identifiers.

    Call this before splitting training and external data.
    """

    data = data.copy()

    for column in [
        coumarin_column,
        cell_line_column,
    ]:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return data


def split_training_external(
    data,
    external_coumarins,
    coumarin_column=GROUP_COLUMN,
):
    """
    Separate training and external observations.

    External coumarins are completely excluded from training.
    """

    data = normalize_identifiers(
        data,
        coumarin_column=coumarin_column,
    )

    external_coumarins = {
        str(coumarin).strip().lower()
        for coumarin in external_coumarins
    }

    external_mask = data[coumarin_column].isin(
        external_coumarins
    )

    training_data = data.loc[
        ~external_mask
    ].copy()

    external_data = data.loc[
        external_mask
    ].copy()

    if training_data.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if external_data.empty:
        raise ValueError(
            "External dataset is empty. Check the compound names."
        )

    overlap = (
        set(training_data[coumarin_column].unique())
        & set(external_data[coumarin_column].unique())
    )

    if overlap:
        raise RuntimeError(
            f"Training/external compound leakage detected: {overlap}"
        )

    return training_data, external_data


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

def run_nested_cv(
    data,
    features,
    estimator,
    param_distributions,
    model_name,
    n_iter=100,
    outer_validation="LOCO",
    n_splits=5,
    target=TARGET_COLUMN,
    group_column=GROUP_COLUMN,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0,
):
    """
    Run nested cross-validation.

    Outer validation:
        KFold: ordinary shuffled K-fold.
        LOCO: Leave-One-Coumarin-Out.

    Inner hyperparameter selection:
        Always Leave-One-Coumarin-Out.

    The estimator should be a complete sklearn estimator or
    Pipeline, and param_distributions must use its parameter names.

    Returns:
        predictions_df
        fold_results_df
        summary_df
    """

    required_columns = (
        list(features)
        + [target, group_column]
    )

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if data[required_columns].isna().any().any():
        raise ValueError(
            "Missing values found in the CV data."
        )

    X = data[features].copy()
    y = data[target].astype(float).to_numpy()

    groups = (
        data[group_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .to_numpy()
    )

    unique_groups = np.unique(groups)

    if len(unique_groups) < 3:
        raise ValueError(
            "At least three distinct coumarins are required "
            "for nested LOCO validation."
        )

    if outer_validation.upper() == "KFold":
        outer_cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        outer_splits = list(
            outer_cv.split(X, y)
        )

    elif outer_validation.upper() == "LOCO":
        outer_cv = LeaveOneGroupOut()

        outer_splits = list(
            outer_cv.split(
                X,
                y,
                groups=groups,
            )
        )

    else:
        raise ValueError(
            "outer_validation must be 'KFold' or 'LOCO'."
        )

    prediction_records = []
    fold_records = []

    for fold_number, (
        train_indices,
        test_indices,
    ) in enumerate(outer_splits, start=1):

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        y_train = y[train_indices]
        y_test = y[test_indices]

        groups_train = groups[train_indices]
        groups_test = groups[test_indices]

        n_inner_groups = len(
            np.unique(groups_train)
        )

        if n_inner_groups < 2:
            raise ValueError(
                "Not enough distinct training coumarins "
                "for inner LOCO validation."
            )

        inner_cv = GroupKFold(
            n_splits=n_inner_groups
        )

        search = RandomizedSearchCV(
            estimator=clone(estimator),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=inner_cv,
            refit=True,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            error_score="raise",
        )

        # Crucially, inner search sees only the outer
        # training observations and their coumarin groups.
        search.fit(
            X_train,
            y_train,
            groups=groups_train,
        )

        predictions = search.predict(X_test)

        metrics = calculate_metrics(
            y_test,
            predictions,
        )

        held_out = (
            str(groups_test[0])
            if outer_validation.upper() == "LOCO"
            else None
        )

        fold_record = {
            "Model": model_name,
            "Validation": outer_validation,
            "Fold": fold_number,
            "HeldOutCoumarin": held_out,
            "BestInnerLOCO_MAE": -search.best_score_,
            **metrics,
            "BestParameters": search.best_params_,
        }

        fold_records.append(fold_record)

        prediction_frame = pd.DataFrame({
            "OriginalIndex": data.index[test_indices],
            "Model": model_name,
            "Validation": outer_validation,
            "Fold": fold_number,
            "Coumarin": groups_test,
            "Actual": y_test,
            "Predicted": predictions,
        })

        prediction_records.append(
            prediction_frame
        )

        held_out_text = (
            f" | Held out: {held_out}"
            if held_out is not None
            else ""
        )

        print(
            f"{model_name} | {outer_validation} "
            f"| Fold {fold_number}"
            f"{held_out_text}"
            f" | Best inner LOCO MAE: {-search.best_score_:.4f}"
            f" | MAE: {metrics['MAE']:.4f}"
            f" | RMSE: {metrics['RMSE']:.4f}"
            f" | R2: {metrics['R2']:.4f}"
        )

    predictions_df = pd.concat(
        prediction_records,
        ignore_index=True,
    )

    fold_results_df = pd.DataFrame(
        fold_records
    )

    overall_metrics = calculate_metrics(
        predictions_df["Actual"],
        predictions_df["Predicted"],
    )

    summary_df = pd.DataFrame([{
        "Model": model_name,
        "Validation": outer_validation,
        **overall_metrics,
    }])

    return (
        predictions_df,
        fold_results_df,
        summary_df,
    )


# ============================================================
# CELL-LINE COVERAGE
# ============================================================

def calculate_training_coverage(
    training_data,
    group_column=GROUP_COLUMN,
    cell_line_column=CELL_LINE_COLUMN,
):
    """
    Count training observations and distinct training coumarins
    for every cell line.

    These counts must be calculated from training data only.
    """

    required = [
        group_column,
        cell_line_column,
    ]

    missing = [
        column
        for column in required
        if column not in training_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = normalize_identifiers(
        training_data,
        coumarin_column=group_column,
        cell_line_column=cell_line_column,
    )

    coverage = (
        data.groupby(cell_line_column)
        .agg(
            TrainingCoumarins=(
                group_column,
                "nunique",
            ),
            TrainingSamples=(
                group_column,
                "size",
            ),
        )
        .reset_index()
    )

    return coverage


# ============================================================
# EXTERNAL EVALUATION
# ============================================================

def evaluate_external_model(
    model,
    external_data,
    features,
    model_name,
    training_data=None,
    thresholds=COVERAGE_THRESHOLDS,
    min_training_coumarins=MIN_TRAINING_COUMARINS,
    target=TARGET_COLUMN,
    group_column=GROUP_COLUMN,
    cell_line_column=CELL_LINE_COLUMN,
):
    """
    Evaluate a fitted model on an external dataset.

    Produces:
        1. Predictions for the full external dataset.
        2. Full external metrics.
        3. Metrics by external coumarin.
        4. Coverage-filtered metrics at each threshold.
        5. Coverage-filtered metrics by coumarin.

    Coverage thresholds affect evaluation subsets only.
    They never alter training or hyperparameter search.

    training_data is optional if coverage filtering is not needed.
    """

    validate_features(
        external_data,
        features,
        target=target,
    )

    external = normalize_identifiers(
        external_data,
        coumarin_column=group_column,
        cell_line_column=cell_line_column,
    )

    X_external = external[features]
    y_external = external[target].astype(float).to_numpy()

    predictions = model.predict(X_external)

    prediction_data = external.copy()

    prediction_data["Model"] = model_name
    prediction_data["Actual"] = y_external
    prediction_data["Predicted"] = predictions
    prediction_data["AbsoluteError"] = np.abs(
        y_external - predictions
    )

    # --------------------------------------------------------
    # Full external metrics
    # --------------------------------------------------------

    full_metrics = pd.DataFrame([{
        "Model": model_name,
        **calculate_metrics(
            y_external,
            predictions,
        ),
    }])

    # --------------------------------------------------------
    # Full external metrics by coumarin
    # --------------------------------------------------------

    coumarin_records = []

    for coumarin, subset in prediction_data.groupby(
        group_column,
        dropna=False,
    ):
        metrics = calculate_metrics(
            subset["Actual"],
            subset["Predicted"],
        )

        coumarin_records.append({
            "Model": model_name,
            "Coumarin": coumarin,
            **metrics,
        })

    by_coumarin = pd.DataFrame(
        coumarin_records
    )

    # --------------------------------------------------------
    # Optional coverage-filtered evaluation
    # --------------------------------------------------------

    coverage_summary_records = []
    coverage_by_coumarin_records = []

    coverage_table = None
    coverage_predictions = None

    if training_data is not None:

        coverage_table = calculate_training_coverage(
            training_data,
            group_column=group_column,
            cell_line_column=cell_line_column,
        )

        # Merge training coverage into external observations.
        # This does not change the fitted model or its predictions.
        coverage_predictions = prediction_data.merge(
            coverage_table,
            on=cell_line_column,
            how="left",
            validate="many_to_one",
        )

        coverage_predictions[
            "TrainingCoumarins"
        ] = coverage_predictions[
            "TrainingCoumarins"
        ].fillna(0).astype(int)

        coverage_predictions[
            "TrainingSamples"
        ] = coverage_predictions[
            "TrainingSamples"
        ].fillna(0).astype(int)

        coverage_predictions[
            "MeetsCoumarinCoverage"
        ] = (
            coverage_predictions["TrainingCoumarins"]
            >= min_training_coumarins
        )

        for threshold in thresholds:

            eligible_mask = (
                coverage_predictions[
                    "MeetsCoumarinCoverage"
                ]
                & (
                    coverage_predictions[
                        "TrainingSamples"
                    ]
                    >= threshold
                )
            )

            subset = coverage_predictions.loc[
                eligible_mask
            ].copy()

            overall_metrics = calculate_metrics(
                subset["Actual"],
                subset["Predicted"],
            )

            coverage_summary_records.append({
                "Model": model_name,
                "Threshold": threshold,
                "MinTrainingCoumarins": min_training_coumarins,
                "NCellLines": subset[
                    cell_line_column
                ].nunique(),
                "NCoumarins": subset[
                    group_column
                ].nunique(),
                **overall_metrics,
            })

            # Keep every external coumarin represented in
            # the output, including compounds with N = 0.
            for coumarin in sorted(
                prediction_data[group_column].dropna().unique()
            ):

                compound_subset = subset.loc[
                    subset[group_column] == coumarin
                ]

                compound_metrics = calculate_metrics(
                    compound_subset["Actual"],
                    compound_subset["Predicted"],
                )

                coverage_by_coumarin_records.append({
                    "Model": model_name,
                    "Threshold": threshold,
                    "Coumarin": coumarin,
                    **compound_metrics,
                })

    coverage_summary = pd.DataFrame(
        coverage_summary_records
    )

    coverage_by_coumarin = pd.DataFrame(
        coverage_by_coumarin_records
    )

    return {
        "predictions": prediction_data,
        "full_metrics": full_metrics,
        "by_coumarin": by_coumarin,
        "training_coverage": coverage_table,
        "coverage_predictions": coverage_predictions,
        "coverage_summary": coverage_summary,
        "coverage_by_coumarin": coverage_by_coumarin,
    }


# ============================================================
# SAVING RESULTS
# ============================================================

def save_model_results(
    results,
    output_directory,
    model_name,
    best_parameters=None,
    fitted_model=None,
):
    """
    Save external evaluation results and optional model artifacts.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_mapping = {
        "predictions": "External_Predictions.csv",
        "full_metrics": "External_Metrics.csv",
        "by_coumarin": "External_By_Coumarin.csv",
        "training_coverage": "Training_CellLine_Coverage.csv",
        "coverage_predictions": "External_Coverage_Predictions.csv",
        "coverage_summary": "External_Coverage_Summary.csv",
        "coverage_by_coumarin": "External_Coverage_By_Coumarin.csv",
    }

    for key, filename in file_mapping.items():
        frame = results.get(key)

        if frame is not None:
            frame.to_csv(
                output_directory / filename,
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
                default=str,
            )

    if fitted_model is not None:
        joblib.dump(
            fitted_model,
            output_directory / f"{model_name}_Final.joblib",
        )

    print(
        f"Results saved to: {output_directory.resolve()}"
    )