from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabulate import tabulate


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "Data"
RESULTS_DIR = PROJECT_ROOT / "Results"

DATA_FILE = DATA_DIR / "Processed_Data.csv"

RANDOM_STATE = 42

TARGET_COLUMN = "Viability"
GROUP_COLUMN = "Coumarin"
CELL_LINE_COLUMN = "CellLine"
AUTHOR_COLUMN = "Author"

DEFAULT_N_SPLITS = 5
DEFAULT_N_ITER = 200
DEFAULT_SCORING = "r2"
DEFAULT_N_JOBS = -1

FORCE_SEARCH = False

# These are upper bounds.
# Actual candidates are restricted using the minimum effective
# numerical rank across the training folds.
PCA_COMPONENTS = {
    "PCA_All_Numerical": range(1, 14),
    "PCA_Descriptors_Fingerprints": range(1, 12),
    "PCA_Fingerprints": range(1, 12),
}

METRIC_SCORING = {
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "MedAE": "neg_median_absolute_error",
}

SCORING_TO_REFIT = {
    "r2": "R2",
    "neg_mean_absolute_error": "MAE",
    "neg_root_mean_squared_error": "RMSE",
    "neg_median_absolute_error": "MedAE",
}


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

MOLECULAR_DESCRIPTOR_FEATURES = [
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
    CELL_LINE_COLUMN,
]

NUMERICAL_DESCRIPTOR_FEATURES = (
    MOLECULAR_DESCRIPTOR_FEATURES + EXPERIMENTAL_FEATURES
)

FEATURE_CONFIGURATIONS = {
    "Descriptors_BaseLine": (
        "Molecular descriptors and experimental features, "
        "without fingerprints."
    ),

    "Descriptor&Morgans_Baseline": (
        "All original features, without PCA."
    ),

    "PCA_All_Numerical": (
        "PCA applied to all numerical features."
    ),

    "PCA_Descriptors_Fingerprints": (
        "PCA applied to molecular descriptors and Morgan fingerprints."
    ),

    "PCA_Fingerprints": (
        "PCA applied only to Morgan fingerprints."
    ),
}

PCA_CONFIGURATIONS = {
    "PCA_All_Numerical",
    "PCA_Descriptors_Fingerprints",
    "PCA_Fingerprints",
}


# ============================================================
# COUMARIN GROUPS
# ============================================================

COUMARIN_GROUPS = {
    "Group_1": [
        "auraptene",
        "umbelliprenin",
        "imperatorin",
    ],

    "Group_2": [
        "umbelliferone",
        "herniarin",
        "scopoletin",
        "esculetin",
        "osthole",
    ],

    "Group_3": [
        "galbanicacid",
        "farnesiferola",
        "farnesiferolb",
        "farnesiferolc",
    ],
}


# ============================================================
# DATA HANDLING
# ============================================================

def normalize_identifiers(data):
    data = data.copy()

    for column in [
        GROUP_COLUMN,
        CELL_LINE_COLUMN,
        AUTHOR_COLUMN,
    ]:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            if column == AUTHOR_COLUMN:
                data[column] = data[column].str.replace(
                    r"\s+",
                    " ",
                    regex=True,
                )

    return data


def load_data(file_path=DATA_FILE):
    data = pd.read_csv(file_path)
    return normalize_identifiers(data)


def get_morgan_features(data):
    return sorted(
        [
            column
            for column in data.columns
            if re.fullmatch(r"Morgan_\d+", str(column))
        ],
        key=lambda column: int(column.split("_")[1]),
    )


def get_morgan_features_from_names(columns):
    return [
        column
        for column in columns
        if re.fullmatch(r"Morgan_\d+", str(column))
    ]


def get_feature_sets(data):
    """Return the raw input columns for every supported configuration."""

    morgan_features = get_morgan_features(data)

    required_features = (
        NUMERICAL_DESCRIPTOR_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing = [
        column
        for column in required_features
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    all_features = (
        NUMERICAL_DESCRIPTOR_FEATURES
        + morgan_features
        + CATEGORICAL_FEATURES
    )

    descriptors_only = (
        NUMERICAL_DESCRIPTOR_FEATURES
        + CATEGORICAL_FEATURES
    )

    return {
        "Descriptors_BaseLine": descriptors_only,

        "Descriptor&Morgans_Baseline": all_features,

        "PCA_All_Numerical": all_features,

        "PCA_Descriptors_Fingerprints": all_features,

        "PCA_Fingerprints": all_features,
    }


def validate_features(data, features):
    required = list(
        dict.fromkeys(
            features
            + [
                TARGET_COLUMN,
                GROUP_COLUMN,
            ]
        )
    )

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    missing_values = data[required].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            "Missing values detected in model data:\n"
            f"{missing_values.to_string()}"
        )

    numerical = [
        column
        for column in features
        if column not in CATEGORICAL_FEATURES
    ]

    non_numeric = [
        column
        for column in numerical
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric:
        raise TypeError(
            "The following numerical features contain non-numeric data: "
            f"{non_numeric}"
        )

    non_finite = [
        column
        for column in numerical
        if not np.isfinite(
            data[column].to_numpy(dtype=float)
        ).all()
    ]

    if non_finite:
        raise ValueError(
            "The following numerical features contain non-finite values: "
            f"{non_finite}"
        )

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "The target must contain at least two distinct values."
        )

    if CELL_LINE_COLUMN not in features:
        raise ValueError(
            "CellLine must always be included as a training feature."
        )

    for feature in EXPERIMENTAL_FEATURES:
        if feature not in features:
            raise ValueError(
                f"{feature} must always be included as a training feature."
            )

    if GROUP_COLUMN in features:
        raise ValueError(
            "Coumarin is a grouping variable, not a training feature."
        )

    if AUTHOR_COLUMN in features:
        raise ValueError(
            "Author is a grouping variable, not a training feature."
        )


# ============================================================
# PARAMETER DISTRIBUTION VALIDATION
# ============================================================

def validate_param_distributions(param_distributions):

    if isinstance(param_distributions, dict):
        distributions = [param_distributions]

    elif isinstance(param_distributions, list):
        distributions = param_distributions

    else:
        raise TypeError(
            "param_distributions must be a dictionary "
            "or a list of dictionaries."
        )

    if not distributions:
        raise ValueError(
            "Parameter distributions cannot be empty."
        )

    for distribution in distributions:

        if not isinstance(distribution, dict):
            raise TypeError(
                "Each parameter distribution must be a dictionary."
            )

        for parameter, values in distribution.items():

            if isinstance(values, (list, tuple, np.ndarray)):

                values = list(values)

                if not values:
                    raise ValueError(
                        f"Empty distribution for {parameter}."
                    )

                try:
                    unique_count = len(set(values))

                except TypeError:
                    unique_count = len({
                        repr(value)
                        for value in values
                    })

                if len(values) != unique_count:
                    raise ValueError(
                        f"Duplicate values detected in parameter "
                        f"'{parameter}': {values}"
                    )

    return distributions


# ============================================================
# PREPROCESSING CONFIGURATION
# ============================================================

def get_preprocessing_configuration(
    data,
    feature_configuration,
):
    feature_sets = get_feature_sets(data)

    if feature_configuration not in feature_sets:
        raise ValueError(
            f"Unknown feature configuration: {feature_configuration}"
        )

    features = feature_sets[feature_configuration]

    morgan_features = get_morgan_features(data)

    if (
        feature_configuration in PCA_CONFIGURATIONS
        or feature_configuration == "Descriptor&Morgans_Baseline"
    ) and not morgan_features:
        raise ValueError(
            f"'{feature_configuration}' requires Morgan fingerprint "
            "columns, but none were found."
        )

    if feature_configuration in {
        "Descriptor&Morgans_Baseline",
        "Descriptors_BaseLine",
    }:
        pca_features = []

    elif feature_configuration == "PCA_All_Numerical":
        pca_features = [
            column
            for column in features
            if column not in CATEGORICAL_FEATURES
        ]

    elif feature_configuration == "PCA_Descriptors_Fingerprints":
        pca_features = (
            MOLECULAR_DESCRIPTOR_FEATURES
            + morgan_features
        )

    elif feature_configuration == "PCA_Fingerprints":
        pca_features = morgan_features

    else:
        raise ValueError(
            f"Unknown feature configuration: {feature_configuration}"
        )

    passthrough_features = [
        column
        for column in features
        if column not in pca_features
        and column not in CATEGORICAL_FEATURES
    ]

    # LogDose and Time must always be represented.
    if not set(EXPERIMENTAL_FEATURES).issubset(features):
        raise ValueError(
            "LogDose and Time must be included in every configuration."
        )

    # CellLine must remain outside PCA.
    if (
        CELL_LINE_COLUMN not in features
        or CELL_LINE_COLUMN in pca_features
    ):
        raise ValueError(
            "CellLine must be included and must remain outside PCA."
        )

    if GROUP_COLUMN in features or AUTHOR_COLUMN in features:
        raise ValueError(
            "Coumarin and Author cannot be training features."
        )

    return (
        features,
        pca_features,
        passthrough_features,
    )


# ============================================================
# PREPROCESSOR AND PIPELINE
# ============================================================

def build_preprocessor(
    features,
    pca_features=None,
    pca_components=None,
):
    pca_features = pca_features or []

    categorical_features = [
        column
        for column in features
        if column in CATEGORICAL_FEATURES
    ]

    passthrough_features = [
        column
        for column in features
        if column not in pca_features
        and column not in categorical_features
    ]

    transformers = []

    if pca_features:

        continuous_pca_features = [
            column
            for column in pca_features
            if column in NUMERICAL_DESCRIPTOR_FEATURES
        ]

        binary_pca_features = (
            get_morgan_features_from_names(pca_features)
        )

        unrecognized_pca_features = [
            column
            for column in pca_features
            if column not in continuous_pca_features
            and column not in binary_pca_features
        ]

        if unrecognized_pca_features:
            raise ValueError(
                "Unsupported columns were assigned to PCA: "
                f"{unrecognized_pca_features}"
            )

        pca_transformers = []

        # Standardize continuous numerical features.
        if continuous_pca_features:
            pca_transformers.append(
                (
                    "continuous",
                    StandardScaler(),
                    continuous_pca_features,
                )
            )

        # Keep Morgan fingerprints binary and unscaled.
        if binary_pca_features:
            pca_transformers.append(
                (
                    "fingerprints",
                    "passthrough",
                    binary_pca_features,
                )
            )

        pca_input_transformer = ColumnTransformer(
            transformers=pca_transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        pca_steps = [
            (
                "prepare_pca_input",
                pca_input_transformer,
            )
        ]

        pca_kwargs = {
            "svd_solver": "full",
            "random_state": RANDOM_STATE,
        }

        if pca_components is not None:
            pca_kwargs["n_components"] = pca_components

        pca_steps.append(
            (
                "pca",
                PCA(**pca_kwargs),
            )
        )

        transformers.append(
            (
                "pca_block",
                Pipeline(steps=pca_steps),
                pca_features,
            )
        )

    if passthrough_features:
        transformers.append(
            (
                "other_numerical",
                "passthrough",
                passthrough_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(
    estimator,
    features,
    pca_features=None,
):
    preprocessor = build_preprocessor(
        features=features,
        pca_features=pca_features,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


# ============================================================
# CROSS-VALIDATION SPLITS
# ============================================================

def get_cv_splits(
    data,
    validation,
    n_splits=DEFAULT_N_SPLITS,
):
    indices = np.arange(len(data))

    if validation == "KFold":

        if n_splits < 2 or n_splits > len(data):
            raise ValueError(
                f"KFold requires 2 <= n_splits <= number of rows. "
                f"Received n_splits={n_splits}, rows={len(data)}."
            )

        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        return list(cv.split(indices))

    if validation in {
        "LOCO",
        "GroupSpecific_LOCO",
    }:

        if data[GROUP_COLUMN].nunique() < 2:
            raise ValueError(
                "LOCO requires at least two distinct coumarins."
            )

        cv = GroupKFold(
            n_splits=data[GROUP_COLUMN].nunique()
        )

        return list(
            cv.split(
                indices,
                groups=data[GROUP_COLUMN],
            )
        )

    if validation == "LOGO":

        if data["CoumarinGroup"].nunique() < 2:
            raise ValueError(
                "LOGO requires at least two coumarin groups."
            )

        cv = GroupKFold(
            n_splits=data["CoumarinGroup"].nunique()
        )

        return list(
            cv.split(
                indices,
                groups=data["CoumarinGroup"],
            )
        )

    raise ValueError(
        f"Unsupported validation strategy: {validation}"
    )


def get_cv(
    data,
    validation,
    n_splits=DEFAULT_N_SPLITS,
):
    if validation == "KFold":
        return KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    if validation in {
        "LOCO",
        "GroupSpecific_LOCO",
    }:
        return GroupKFold(
            n_splits=data[GROUP_COLUMN].nunique()
        )

    if validation == "LOGO":
        return GroupKFold(
            n_splits=data["CoumarinGroup"].nunique()
        )

    raise ValueError(
        f"Unsupported validation strategy: {validation}"
    )


# ============================================================
# PCA COMPONENT CANDIDATES
# ============================================================

def get_pca_input_matrix(
    X_train,
    pca_features,
):
    """
    Prepare PCA input using training data only.

    Continuous variables are standardized.
    Morgan fingerprints remain binary and unscaled.
    """

    continuous_features = [
        column
        for column in pca_features
        if column in NUMERICAL_DESCRIPTOR_FEATURES
    ]

    binary_features = (
        get_morgan_features_from_names(pca_features)
    )

    matrices = []

    if continuous_features:
        scaler = StandardScaler()

        matrices.append(
            scaler.fit_transform(
                X_train[continuous_features]
            )
        )

    if binary_features:
        matrices.append(
            X_train[binary_features].to_numpy(dtype=float)
        )

    if not matrices:
        raise ValueError(
            "No features were provided for PCA."
        )

    return np.concatenate(
        matrices,
        axis=1,
    )


def get_effective_pca_rank(
    X_train,
    pca_features,
):
    """
    Calculate effective PCA rank after centering.

    PCA centers its input internally, so the rank is calculated
    from the centered matrix rather than the uncentered matrix.
    """

    matrix = get_pca_input_matrix(
        X_train,
        pca_features,
    )

    centered_matrix = (
        matrix - matrix.mean(axis=0)
    )

    return int(
        np.linalg.matrix_rank(centered_matrix)
    )


def get_pca_component_candidates(
    data,
    features,
    pca_features,
    validation="KFold",
    n_splits=DEFAULT_N_SPLITS,
    candidates=None,
):
    """
    Determine valid PCA dimensions using the minimum effective
    rank across all training folds.
    """

    if not pca_features:
        return [], {}

    if candidates is None:
        candidates = range(
            1,
            len(pca_features) + 1,
        )

    splits = get_cv_splits(
        data,
        validation,
        n_splits,
    )

    fold_ranks = []

    for fold_number, (train_indices, _) in enumerate(
        splits,
        start=1,
    ):

        X_train = data.iloc[train_indices][features]

        rank = get_effective_pca_rank(
            X_train,
            pca_features,
        )

        fold_ranks.append({
            "Fold": fold_number,
            "TrainingSamples": len(train_indices),
            "EffectiveRank": rank,
        })

    minimum_rank = min(
        row["EffectiveRank"]
        for row in fold_ranks
    )

    minimum_training_size = min(
        row["TrainingSamples"]
        for row in fold_ranks
    )

    # PCA centers its input, so the maximum meaningful rank
    # cannot exceed n_samples - 1.
    maximum_components = min(
        minimum_rank,
        minimum_training_size - 1,
    )

    valid_candidates = sorted({
        int(component)
        for component in candidates
        if isinstance(
            component,
            (int, np.integer),
        )
        and not isinstance(
            component,
            (bool, np.bool_),
        )
        and 1 <= int(component) <= maximum_components
    })

    if not valid_candidates:
        raise ValueError(
            "No valid PCA component candidates.\n"
            f"Minimum effective rank: {minimum_rank}\n"
            f"Maximum allowed components: {maximum_components}"
        )

    information = {
        "PCAFeatures": pca_features,
        "FoldRanks": fold_ranks,
        "MinimumEffectiveRank": minimum_rank,
        "MaximumComponents": maximum_components,
        "Candidates": valid_candidates,
    }

    return valid_candidates, information


# ============================================================
# SEARCH DISTRIBUTIONS
# ============================================================

def build_search_distributions(
    param_distributions,
    pca_candidates=None,
):
    distributions = validate_param_distributions(
        param_distributions
    )

    if pca_candidates is None:
        return distributions

    pca_parameter = (
        "preprocessor__pca_block__pca__n_components"
    )

    search_distributions = []

    for distribution in distributions:

        if pca_parameter in distribution:
            raise ValueError(
                "PCA component count is managed by model_utils2.py. "
                "Do not include it in the model script."
            )

        updated_distribution = distribution.copy()

        updated_distribution[pca_parameter] = list(
            pca_candidates
        )

        search_distributions.append(
            updated_distribution
        )

    return search_distributions


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MedAE": median_absolute_error(y_true, y_pred),
        "R2": (
            r2_score(y_true, y_pred)
            if len(y_true) >= 2
            else np.nan
        ),
    }


def get_fold_results(
    cv_results,
    validation,
    data,
    n_splits=DEFAULT_N_SPLITS,
    best_index=0,
):
    """
    Extract fold-level scores for the best hyperparameter combination.

    Error metrics are converted back to positive values.
    """

    splits = get_cv_splits(
        data,
        validation,
        n_splits,
    )

    rows = []

    for fold_number, (_, test_indices) in enumerate(splits):

        row = {
            "Fold": fold_number + 1,
            "Validation": validation,
            "N_Test": len(test_indices),
        }

        if validation in {
            "LOCO",
            "GroupSpecific_LOCO",
        }:
            row["HeldOutCoumarin"] = (
                data.iloc[test_indices][GROUP_COLUMN]
                .iloc[0]
            )

        elif validation == "LOGO":
            row["HeldOutGroup"] = (
                data.iloc[test_indices]["CoumarinGroup"]
                .iloc[0]
            )

            row["HeldOutCoumarins"] = ", ".join(
                sorted(
                    data.iloc[test_indices][GROUP_COLUMN]
                    .unique()
                )
            )

        for metric in METRIC_SCORING:

            column = f"split{fold_number}_test_{metric}"

            if column not in cv_results.columns:
                raise KeyError(
                    f"Missing CV result column: {column}"
                )

            value = float(
                cv_results.loc[best_index, column]
            )

            if metric != "R2":
                value = -value

            row[metric] = value

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_fixed_parameters(
    estimator,
    X,
    y,
    cv,
    best_parameters,
    groups=None,
    scoring=None,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Re-evaluate cached hyperparameters using cross-validation.

    Returns a DataFrame with the same general structure as
    RandomizedSearchCV.cv_results_.
    """

    evaluation = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        groups=groups,
        scoring=scoring or METRIC_SCORING,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
    )

    results = {
        "params": [best_parameters],
    }

    for metric in METRIC_SCORING:

        test_scores = evaluation[
            f"test_{metric}"
        ]

        for fold_number, score in enumerate(
            test_scores
        ):
            results[
                f"split{fold_number}_test_{metric}"
            ] = [score]

        results[f"mean_test_{metric}"] = [
            np.mean(test_scores)
        ]

        results[f"std_test_{metric}"] = [
            np.std(test_scores)
        ]

        results[f"rank_test_{metric}"] = [1]

        train_scores = evaluation.get(
            f"train_{metric}"
        )

        if train_scores is not None:
            results[f"mean_train_{metric}"] = [
                np.mean(train_scores)
            ]

            results[f"std_train_{metric}"] = [
                np.std(train_scores)
            ]

    return pd.DataFrame(results)


# ============================================================
# SEARCH AND CACHE HELPERS
# ============================================================

def get_experiment_directory(
    model_name,
    feature_configuration,
    validation,
    group_name=None,
):
    path = (
        RESULTS_DIR
        / "FeatureEngineering"
        / model_name
        / feature_configuration
        / validation
    )

    if group_name is not None:
        path = path / group_name

    return path


def get_best_parameters_path(
    model_name,
    feature_configuration,
    validation,
    group_name=None,
):
    return (
        get_experiment_directory(
            model_name,
            feature_configuration,
            validation,
            group_name,
        )
        / "BestParameters.json"
    )


def load_best_parameters(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        saved = json.load(file)

    # Support both the old direct-dictionary format
    # and the newer wrapped format.
    if "BestParameters" in saved:
        return saved["BestParameters"]

    return saved


def save_best_parameters(
    parameters,
    path,
    best_score=None,
    scoring=None,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "BestParameters": parameters,
        "BestScore": best_score,
        "Scoring": scoring,
    }

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


def build_randomized_search(
    pipeline,
    distributions,
    cv,
    scoring,
    n_iter,
    n_jobs,
):
    if scoring not in SCORING_TO_REFIT:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. "
            f"Choose from {list(SCORING_TO_REFIT)}."
        )

    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=distributions,
        n_iter=n_iter,
        scoring=METRIC_SCORING,
        refit=SCORING_TO_REFIT[scoring],
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=True,
        verbose=1,
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value


# ============================================================
# SHARED CROSS-VALIDATION EXPERIMENT
# ============================================================

def _run_cv_experiment(
    data,
    estimator,
    param_distributions,
    feature_configuration,
    validation,
    model_name,
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
    group_name=None,
):
    data = normalize_identifiers(
        data
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError(
            "The input dataset is empty."
        )

    if validation == "LOGO":

        coumarin_to_group = {
            coumarin: group
            for group, coumarins in COUMARIN_GROUPS.items()
            for coumarin in coumarins
        }

        data["CoumarinGroup"] = (
            data[GROUP_COLUMN].map(coumarin_to_group)
        )

        if data["CoumarinGroup"].isna().any():
            unknown = sorted(
                data.loc[
                    data["CoumarinGroup"].isna(),
                    GROUP_COLUMN,
                ].unique()
            )

            raise ValueError(
                "Some coumarins are not assigned to a group: "
                f"{unknown}"
            )

    (
        features,
        pca_features,
        _,
    ) = get_preprocessing_configuration(
        data,
        feature_configuration,
    )

    validate_features(
        data,
        features,
    )

    X = data[features].copy()
    y = data[TARGET_COLUMN].astype(float)

    cv = get_cv(
        data,
        validation,
        n_splits,
    )

    groups = None

    if validation in {
        "LOCO",
        "GroupSpecific_LOCO",
    }:
        groups = data[GROUP_COLUMN].to_numpy()

    elif validation == "LOGO":
        groups = data["CoumarinGroup"].to_numpy()

    # --------------------------------------------------------
    # PCA candidates
    # --------------------------------------------------------

    if pca_features:

        pca_candidates, pca_information = (
            get_pca_component_candidates(
                data=data,
                features=features,
                pca_features=pca_features,
                validation=validation,
                n_splits=n_splits,
                candidates=PCA_COMPONENTS[
                    feature_configuration
                ],
            )
        )

    else:
        pca_candidates = None
        pca_information = {}

    search_distributions = build_search_distributions(
        param_distributions,
        pca_candidates,
    )

    # --------------------------------------------------------
    # Search/cache
    # --------------------------------------------------------

    parameter_path = get_best_parameters_path(
        model_name,
        feature_configuration,
        validation,
        group_name,
    )

    search = None

    cached_parameters = None

    if parameter_path.exists() and not FORCE_SEARCH:
        cached_parameters = load_best_parameters(
            parameter_path
        )

    pipeline = build_pipeline(
        estimator,
        features,
        pca_features,
    )

    if cached_parameters is not None:

        print(
            "\nLoading previously saved best parameters..."
        )

        best_parameters = cached_parameters

        pipeline.set_params(
            **best_parameters
        )

        print("\nLoaded Best Parameters:")

        print(
            json.dumps(
                _json_safe(best_parameters),
                indent=4,
            )
        )

        print(
            "\nEvaluating cached parameters using cross-validation..."
        )

        raw_cv_results = evaluate_fixed_parameters(
            estimator=pipeline,
            X=X,
            y=y,
            cv=cv,
            best_parameters=best_parameters,
            groups=groups,
            scoring=METRIC_SCORING,
            n_jobs=n_jobs,
        )

        best_index = 0

        best_score = float(
            raw_cv_results.loc[
                best_index,
                f"mean_test_{SCORING_TO_REFIT[scoring]}",
            ]
        )

        best_model = pipeline.fit(
            X,
            y,
        )

        search_performed = False

    else:

        print(
            f"\nStarting Randomized Search "
            f"({n_iter} iterations)..."
        )

        search = build_randomized_search(
            pipeline=pipeline,
            distributions=search_distributions,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )

        if groups is None:
            search.fit(X, y)
        else:
            search.fit(
                X,
                y,
                groups=groups,
            )

        best_model = search.best_estimator_
        best_parameters = search.best_params_
        best_score = float(search.best_score_)
        best_index = search.best_index_

        raw_cv_results = pd.DataFrame(
            search.cv_results_
        )

        save_best_parameters(
            parameters=best_parameters,
            path=parameter_path,
            best_score=best_score,
            scoring=scoring,
        )

        search_performed = True

    # --------------------------------------------------------
    # Display best parameters and results
    # --------------------------------------------------------

    refit_metric = SCORING_TO_REFIT[scoring]

    print("\n" + "=" * 80)
    print("BEST RESULTS")
    print("=" * 80)

    print(
        f"Best {refit_metric} CV score: {best_score:.6f}"
    )

    print("\nBest Parameters:")

    print(
        json.dumps(
            _json_safe(best_parameters),
            indent=4,
        )
    )

    # --------------------------------------------------------
    # Fold results
    # --------------------------------------------------------

    fold_results = get_fold_results(
        cv_results=raw_cv_results,
        validation=validation,
        data=data,
        n_splits=n_splits,
        best_index=best_index,
    )

    # --------------------------------------------------------
    # Feature counts
    # --------------------------------------------------------

    transformed_features = (
        best_model.named_steps["preprocessor"].transform(X)
    )

    transformed_feature_count = (
        transformed_features.shape[1]
    )

    original_feature_count = (
        len([
            feature
            for feature in features
            if feature != CELL_LINE_COLUMN
        ])
        + data[CELL_LINE_COLUMN].nunique()
    )

    raw_input_column_count = len(features)

    reduction_percentage = (
        (
            original_feature_count
            - transformed_feature_count
        )
        / original_feature_count
        * 100
    )

    # --------------------------------------------------------
    # PCA information
    # --------------------------------------------------------

    pca_information.update({
        "PCA_Components": None,
        "PCA_Explained_Variance": None,
        "PCA_Explained_Variance_Ratio": [],
        "PCA_Cumulative_Explained_Variance": [],
    })

    if pca_features:

        pca = (
            best_model
            .named_steps["preprocessor"]
            .named_transformers_["pca_block"]
            .named_steps["pca"]
        )

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        pca_information.update({
            "PCA_Components": int(pca.n_components_),
            "PCA_Explained_Variance": float(
                explained.sum()
            ),
            "PCA_Explained_Variance_Ratio": (
                explained.tolist()
            ),
            "PCA_Cumulative_Explained_Variance": (
                cumulative.tolist()
            ),
        })

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "Model": model_name,
        "FeatureConfiguration": feature_configuration,
        "Validation": validation,
        "N": len(data),

        "OriginalFeatures": original_feature_count,
        "RawInputColumns": raw_input_column_count,
        "TransformedFeatures": transformed_feature_count,

        "FeatureReductionPercentage": reduction_percentage,

        "BestMetric": refit_metric,
        "BestCVScore": best_score,

        "BestCVR2": float(
            raw_cv_results.loc[
                best_index,
                "mean_test_R2",
            ]
        ),

        "MeanCV_MAE": float(
            -raw_cv_results.loc[
                best_index,
                "mean_test_MAE",
            ]
        ),

        "MeanCV_RMSE": float(
            -raw_cv_results.loc[
                best_index,
                "mean_test_RMSE",
            ]
        ),

        "MeanCV_MedAE": float(
            -raw_cv_results.loc[
                best_index,
                "mean_test_MedAE",
            ]
        ),

        "BestParameters": str(best_parameters),

        "PCA_Components": pca_information[
            "PCA_Components"
        ],

        "PCA_Explained_Variance": pca_information[
            "PCA_Explained_Variance"
        ],
    }

    summary = pd.DataFrame([summary])

    cv_results = (
        raw_cv_results
        .sort_values(
            f"rank_test_{refit_metric}"
        )
        .reset_index(drop=True)
    )

    results = {
        "Summary": summary,
        "CVResults": cv_results,
        "FoldResults": fold_results,
    }

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\nSummary:")
    print_dataframe(summary)

    print("\nFold Results:")
    print_dataframe(fold_results)

    if search_performed:

        print("\nTop Search Results:")

        display_columns = [
            f"rank_test_{refit_metric}",
            "mean_test_R2",
            "mean_test_MAE",
            "mean_test_RMSE",
            "mean_test_MedAE",
            "params",
        ]

        print_dataframe(
            cv_results.head(10)[display_columns]
        )

    else:
        print(
            "\nHyperparameter search skipped: "
            "using cached best parameters."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "ModelName": model_name,
        "FeatureConfiguration": feature_configuration,
        "Validation": validation,
        "GroupName": group_name,

        "BestModel": best_model,
        "BestParameters": best_parameters,
        "BestScore": best_score,

        "Results": results,

        "PCAInformation": pca_information,

        "FeatureCount": {
            "Original": original_feature_count,
            "RawInputColumns": raw_input_column_count,
            "Transformed": transformed_feature_count,
            "ReductionPercentage": reduction_percentage,
            "PCAInput": len(pca_features),
        },

        "Search": search,
        "SearchPerformed": search_performed,

        "Features": features,
        "SampleCount": len(data),
    }


# ============================================================
# EXPERIMENT 1: KFOLD
# ============================================================

def run_kfold_experiment(
    data,
    estimator,
    param_distributions,
    model_name,
    feature_configuration,
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    return _run_cv_experiment(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        feature_configuration=feature_configuration,
        validation="KFold",
        model_name=model_name,
        n_splits=n_splits,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
    )


# ============================================================
# EXPERIMENT 2: LOCO
# ============================================================

def run_loco_experiment(
    data,
    estimator,
    param_distributions,
    model_name,
    feature_configuration,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    return _run_cv_experiment(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        feature_configuration=feature_configuration,
        validation="LOCO",
        model_name=model_name,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
    )


# ============================================================
# EXPERIMENT 3: LOGO
# ============================================================

def run_logo_experiment(
    data,
    estimator,
    param_distributions,
    model_name,
    feature_configuration,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Leave One Coumarin Group Out.

    Each fold leaves out one predefined group of coumarins.
    """

    return _run_cv_experiment(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        feature_configuration=feature_configuration,
        validation="LOGO",
        model_name=model_name,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
    )


# ============================================================
# EXPERIMENT 4: GROUP-SPECIFIC LOCO
# ============================================================

def run_group_specific_experiment(
    data,
    estimator,
    param_distributions,
    model_name,
    feature_configuration,
    group_name,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Restrict the dataset to one predefined group,
    then perform LOCO within that group.
    """

    if group_name not in COUMARIN_GROUPS:
        raise ValueError(
            f"Unknown group '{group_name}'. "
            f"Choose from {list(COUMARIN_GROUPS)}."
        )

    data = normalize_identifiers(data)

    selected_coumarins = COUMARIN_GROUPS[group_name]

    group_data = data[
        data[GROUP_COLUMN].isin(selected_coumarins)
    ].copy()

    if group_data.empty:
        raise ValueError(
            f"No samples found for {group_name}."
        )

    if group_data[GROUP_COLUMN].nunique() < 2:
        raise ValueError(
            f"{group_name} contains fewer than two coumarins. "
            "LOCO cannot be performed."
        )

    return _run_cv_experiment(
        data=group_data,
        estimator=estimator,
        param_distributions=param_distributions,
        feature_configuration=feature_configuration,
        validation="GroupSpecific_LOCO",
        model_name=model_name,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        group_name=group_name,
    )


# ============================================================
# EXPERIMENT 5: AUTHOR HOLDOUT
# ============================================================

def run_author_holdout_experiment(
    data,
    estimator,
    param_distributions,
    model_name,
    feature_configuration,
    test_size=0.10,
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Split by Author:

        90% training authors
        10% unseen test authors

    Hyperparameter search uses only the training authors.

    Inner CV also uses GroupKFold by Author.

    The test authors are evaluated only once, after tuning.
    """

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    data = normalize_identifiers(
        data
    ).reset_index(drop=True)

    if data[AUTHOR_COLUMN].isna().any():
        raise ValueError(
            "Author contains missing values. "
            "Resolve them before performing the author holdout."
        )

    if data[AUTHOR_COLUMN].nunique() < 3:
        raise ValueError(
            "Author holdout requires at least three distinct authors."
        )

    (
        features,
        pca_features,
        _,
    ) = get_preprocessing_configuration(
        data,
        feature_configuration,
    )

    validate_features(
        data,
        features,
    )

    X = data[features].copy()
    y = data[TARGET_COLUMN].astype(float)

    authors = data[AUTHOR_COLUMN].to_numpy()

    # --------------------------------------------------------
    # 90/10 group split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=authors,
        )
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    authors_train = authors[train_indices]
    authors_test = authors[test_indices]

    train_authors = sorted(
        pd.unique(authors_train).tolist()
    )

    test_authors = sorted(
        pd.unique(authors_test).tolist()
    )

    # Explicit leakage check.
    author_overlap = (
        set(train_authors)
        & set(test_authors)
    )

    if author_overlap:
        raise RuntimeError(
            "Author leakage detected between training and testing: "
            f"{sorted(author_overlap)}"
        )

    # --------------------------------------------------------
    # Inner grouped CV
    # --------------------------------------------------------

    number_of_train_authors = len(train_authors)

    if number_of_train_authors < 2:
        raise ValueError(
            "The training split contains fewer than two authors."
        )

    actual_splits = min(
        n_splits,
        number_of_train_authors,
    )

    cv = GroupKFold(
        n_splits=actual_splits
    )

    inner_splits = list(
        cv.split(
            X_train,
            y_train,
            groups=authors_train,
        )
    )

    # --------------------------------------------------------
    # PCA candidates based on the actual inner grouped folds
    # --------------------------------------------------------

    if pca_features:

        fold_ranks = []

        for fold_number, (inner_train, _) in enumerate(
            inner_splits,
            start=1,
        ):

            X_inner_train = X_train.iloc[inner_train]

            rank = get_effective_pca_rank(
                X_inner_train,
                pca_features,
            )

            fold_ranks.append({
                "Fold": fold_number,
                "TrainingSamples": len(inner_train),
                "EffectiveRank": rank,
            })

        minimum_rank = min(
            row["EffectiveRank"]
            for row in fold_ranks
        )

        minimum_training_size = min(
            row["TrainingSamples"]
            for row in fold_ranks
        )

        maximum_components = min(
            minimum_rank,
            minimum_training_size - 1,
        )

        pca_candidates = [
            int(component)
            for component in PCA_COMPONENTS[
                feature_configuration
            ]
            if component <= maximum_components
        ]

        if not pca_candidates:
            raise ValueError(
                "No valid PCA candidates for author-grouped CV. "
                f"Maximum components: {maximum_components}"
            )

        pca_information = {
            "PCAFeatures": pca_features,
            "FoldRanks": fold_ranks,
            "MinimumEffectiveRank": minimum_rank,
            "MaximumComponents": maximum_components,
            "Candidates": pca_candidates,
        }

    else:
        pca_candidates = None
        pca_information = {}

    distributions = build_search_distributions(
        param_distributions,
        pca_candidates,
    )

    # --------------------------------------------------------
    # Search/cache
    # --------------------------------------------------------

    validation = "AuthorHoldout_90_10"

    parameter_path = get_best_parameters_path(
        model_name,
        feature_configuration,
        validation,
    )

    cached_parameters = None

    if parameter_path.exists() and not FORCE_SEARCH:
        cached_parameters = load_best_parameters(
            parameter_path
        )

    pipeline = build_pipeline(
        estimator,
        features,
        pca_features,
    )

    search = None

    if cached_parameters is None:

        print(
            "\nStarting author-grouped hyperparameter search..."
        )

        search = build_randomized_search(
            pipeline=pipeline,
            distributions=distributions,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )

        search.fit(
            X_train,
            y_train,
            groups=authors_train,
        )

        best_parameters = search.best_params_
        best_score = float(search.best_score_)

        raw_cv_results = pd.DataFrame(
            search.cv_results_
        )

        save_best_parameters(
            parameters=best_parameters,
            path=parameter_path,
            best_score=best_score,
            scoring=scoring,
        )

        search_performed = True

    else:

        print(
            "\nLoading cached author-grouped parameters..."
        )

        best_parameters = cached_parameters

        pipeline.set_params(
            **best_parameters
        )

        raw_cv_results = evaluate_fixed_parameters(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            best_parameters=best_parameters,
            groups=authors_train,
            scoring=METRIC_SCORING,
            n_jobs=n_jobs,
        )

        best_score = float(
            raw_cv_results.loc[
                0,
                f"mean_test_{SCORING_TO_REFIT[scoring]}",
            ]
        )

        search_performed = False

    # --------------------------------------------------------
    # Fit on training authors only
    # --------------------------------------------------------

    best_model = build_pipeline(
        estimator,
        features,
        pca_features,
    )

    best_model.set_params(
        **best_parameters
    )

    best_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate unseen authors exactly once
    # --------------------------------------------------------

    predictions = best_model.predict(
        X_test
    )

    holdout_metrics = calculate_metrics(
        y_test,
        predictions,
    )

    print("\nBest Parameters:")
    print(
        json.dumps(
            _json_safe(best_parameters),
            indent=4,
        )
    )

    print("\nAuthor Holdout Results (10% unseen authors):")
    print_dataframe(
        pd.DataFrame([holdout_metrics])
    )

    # --------------------------------------------------------
    # Test prediction details
    # --------------------------------------------------------

    test_predictions = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions,
        "Author": data.iloc[test_indices][AUTHOR_COLUMN].to_numpy(),
        "Coumarin": data.iloc[test_indices][GROUP_COLUMN].to_numpy(),
        "CellLine": data.iloc[test_indices][CELL_LINE_COLUMN].to_numpy(),
    })

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {
        "ModelName": model_name,
        "FeatureConfiguration": feature_configuration,
        "Validation": validation,

        "BestModel": best_model,
        "BestParameters": best_parameters,
        "BestScore": best_score,

        "Summary": holdout_metrics,
        "HoldoutMetrics": holdout_metrics,

        "SearchCVResults": raw_cv_results,

        "PCAInformation": pca_information,

        "Features": features,

        "Search": search,
        "SearchPerformed": search_performed,

        "TrainSamples": len(train_indices),
        "TestSamples": len(test_indices),

        "TrainAuthors": train_authors,
        "TestAuthors": test_authors,

        "TestPredictions": test_predictions,
    }


# ============================================================
# BACKWARD-COMPATIBLE EXPERIMENT FUNCTION
# ============================================================

def run_experiment(
    data,
    estimator,
    param_distributions,
    feature_configuration="Descriptors_BaseLine",
    validation="KFold",
    model_name="Model",
    n_splits=DEFAULT_N_SPLITS,
    n_iter=DEFAULT_N_ITER,
    scoring=DEFAULT_SCORING,
    n_jobs=DEFAULT_N_JOBS,
):
    """
    Backward-compatible interface for existing model scripts.

    New scripts should preferably use the explicit experiment
    functions: run_kfold_experiment, run_loco_experiment, etc.
    """

    if validation not in {
        "KFold",
        "LOCO",
    }:
        raise ValueError(
            "The compatibility run_experiment() supports "
            "only KFold and LOCO. Use the dedicated functions "
            "for LOGO, group-specific, or author holdout."
        )

    return _run_cv_experiment(
        data=data,
        estimator=estimator,
        param_distributions=param_distributions,
        feature_configuration=feature_configuration,
        validation=validation,
        model_name=model_name,
        n_splits=n_splits,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_experiment_results(
    experiment,
    model_name=None,
    feature_configuration=None,
    validation=None,
    output_directory=None,
):
    model_name = (
        model_name
        or experiment["ModelName"]
    )

    feature_configuration = (
        feature_configuration
        or experiment["FeatureConfiguration"]
    )

    validation = (
        validation
        or experiment["Validation"]
    )

    group_name = experiment.get("GroupName")

    if output_directory is None:
        output_directory = get_experiment_directory(
            model_name,
            feature_configuration,
            validation,
            group_name,
        )

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Cross-validation experiments
    # --------------------------------------------------------

    if "Results" in experiment:

        for name, dataframe in experiment["Results"].items():

            output_file = (
                output_directory / f"{name}.csv"
            )

            # A cached evaluation must not overwrite the
            # original RandomizedSearchCV results.
            if (
                name == "CVResults"
                and not experiment.get(
                    "SearchPerformed",
                    False,
                )
                and output_file.exists()
            ):
                continue

            dataframe.to_csv(
                output_file,
                index=False,
            )

    # --------------------------------------------------------
    # Author holdout experiment
    # --------------------------------------------------------

    if validation == "AuthorHoldout_90_10":

        experiment["TestPredictions"].to_csv(
            output_directory / "TestPredictions.csv",
            index=False,
        )

        pd.DataFrame({
            "Author": experiment["TrainAuthors"],
        }).to_csv(
            output_directory / "TrainAuthors.csv",
            index=False,
        )

        pd.DataFrame({
            "Author": experiment["TestAuthors"],
        }).to_csv(
            output_directory / "TestAuthors.csv",
            index=False,
        )

        pd.DataFrame([
            experiment["HoldoutMetrics"]
        ]).to_csv(
            output_directory / "Summary.csv",
            index=False,
        )

        cv_file = output_directory / "CVResults.csv"

        if (
            experiment["SearchPerformed"]
            or not cv_file.exists()
        ):
            experiment["SearchCVResults"].to_csv(
                cv_file,
                index=False,
            )

    # --------------------------------------------------------
    # Best parameters
    # --------------------------------------------------------

    with open(
        output_directory / "BestParameters.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(
                experiment["BestParameters"]
            ),
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "Model": model_name,
        "FeatureConfiguration": feature_configuration,
        "Validation": validation,

        "BestScore": experiment.get("BestScore"),

        "FeatureCount": experiment.get("FeatureCount"),

        "PCAInformation": experiment.get("PCAInformation"),

        "Features": experiment.get("Features"),

        "GroupName": group_name,

        "TrainSamples": experiment.get("TrainSamples"),
        "TestSamples": experiment.get("TestSamples"),

        "TrainAuthors": experiment.get("TrainAuthors"),
        "TestAuthors": experiment.get("TestAuthors"),
    }

    with open(
        output_directory / "Metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            _json_safe(metadata),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(
        f"\nResults saved to: {output_directory}"
    )

    return output_directory


# ============================================================
# PLOTTING
# ============================================================

def plot_model_results(
    model_name,
    results_directory=None,
):
    """
    Load saved experiment results and generate comparison plots.

    KFold, LOCO, LOGO, group-specific LOCO, and author holdout
    are treated as separate validation strategies.
    """

    base_directory = (
        Path(results_directory)
        if results_directory is not None
        else RESULTS_DIR / "FeatureEngineering" / model_name
    )

    plot_directory = base_directory / "Plots"

    plot_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    configurations = list(
        FEATURE_CONFIGURATIONS.keys()
    )

    summaries = []
    fold_frames = []
    metadata_records = {}

    # --------------------------------------------------------
    # Load saved results
    # --------------------------------------------------------

    for configuration in configurations:

        configuration_directory = (
            base_directory / configuration
        )

        if not configuration_directory.exists():
            continue

        for summary_path in configuration_directory.rglob(
            "Summary.csv"
        ):

            run_directory = summary_path.parent

            try:
                summary = pd.read_csv(
                    summary_path
                )

            except pd.errors.EmptyDataError:
                continue

            if summary.empty:
                continue

            metadata_path = (
                run_directory / "Metadata.json"
            )

            if metadata_path.exists():
                with open(
                    metadata_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    metadata = json.load(file)
            else:
                metadata = {}

            validation = metadata.get(
                "Validation",
                run_directory.name,
            )

            group_name = metadata.get(
                "GroupName",
                run_directory.name
                if run_directory.name.startswith("Group_")
                else None,
            )

            summary["Model"] = model_name
            summary["FeatureConfiguration"] = configuration
            summary["Validation"] = validation
            summary["GroupName"] = group_name

            summaries.append(summary)

            folds_path = (
                run_directory / "FoldResults.csv"
            )

            if folds_path.exists():

                folds = pd.read_csv(
                    folds_path
                )

                if not folds.empty:
                    folds["Model"] = model_name
                    folds["FeatureConfiguration"] = configuration
                    folds["Validation"] = validation
                    folds["GroupName"] = group_name

                    fold_frames.append(folds)

            metadata_records[
                (
                    configuration,
                    validation,
                    group_name,
                )
            ] = metadata

    if not summaries:
        raise FileNotFoundError(
            f"No experiment summaries were found under {base_directory}."
        )

    summary_data = pd.concat(
        summaries,
        ignore_index=True,
    )

    folds_data = (
        pd.concat(
            fold_frames,
            ignore_index=True,
        )
        if fold_frames
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Identify metric columns
    # --------------------------------------------------------

    metric_columns = {
        "R2": [
            "BestCVR2",
            "R2",
            "MeanCV_R2",
        ],

        "MAE": [
            "MeanCV_MAE",
            "MAE",
        ],

        "RMSE": [
            "MeanCV_RMSE",
            "RMSE",
        ],

        "MedAE": [
            "MeanCV_MedAE",
            "MedAE",
        ],
    }

    metric_labels = {
        "R2": "R²",
        "MAE": "MAE",
        "RMSE": "RMSE",
        "MedAE": "Median AE",
    }

    for metric, candidates in metric_columns.items():

        available_column = next(
            (
                column
                for column in candidates
                if column in summary_data.columns
            ),
            None,
        )

        if available_column is not None:
            summary_data[f"Plot_{metric}"] = pd.to_numeric(
                summary_data[available_column],
                errors="coerce",
            )

        else:
            summary_data[f"Plot_{metric}"] = np.nan

    # --------------------------------------------------------
    # Fill missing summary metrics from fold results
    # --------------------------------------------------------

    if not folds_data.empty:

        for metric in metric_labels:

            if metric not in folds_data.columns:
                continue

            grouped = folds_data.groupby(
                [
                    "FeatureConfiguration",
                    "Validation",
                    "GroupName",
                ],
                dropna=False,
            )[metric].mean()

            for index, row in summary_data.iterrows():

                key = (
                    row["FeatureConfiguration"],
                    row["Validation"],
                    row["GroupName"],
                )

                if pd.isna(
                    summary_data.loc[
                        index,
                        f"Plot_{metric}",
                    ]
                ) and key in grouped.index:

                    summary_data.loc[
                        index,
                        f"Plot_{metric}",
                    ] = grouped.loc[key]

    # --------------------------------------------------------
    # Plot 1: Performance by validation strategy
    # --------------------------------------------------------

    for validation, subset in summary_data.groupby(
        "Validation",
        dropna=False,
    ):

        for metric, title in metric_labels.items():

            if subset[f"Plot_{metric}"].isna().all():
                continue

            labels = (
                subset["FeatureConfiguration"].astype(str)
            )

            if subset["GroupName"].notna().any():
                labels = (
                    labels
                    + "\n"
                    + subset["GroupName"].fillna("").astype(str)
                )

            fig, ax = plt.subplots(
                figsize=(max(10, len(subset) * 1.5), 6)
            )

            ax.bar(
                labels,
                subset[f"Plot_{metric}"],
            )

            ax.set_title(
                f"{model_name}: {title} — {validation}"
            )

            ax.set_ylabel(title)
            ax.tick_params(
                axis="x",
                rotation=30,
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            fig.tight_layout()

            fig.savefig(
                plot_directory
                / f"{validation}_{metric}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    # --------------------------------------------------------
    # Plot 2: Fold-level R2 distributions
    # --------------------------------------------------------

    if not folds_data.empty and "R2" in folds_data.columns:

        cv_folds = folds_data[
            folds_data["Validation"].isin([
                "KFold",
                "LOCO",
                "LOGO",
                "GroupSpecific_LOCO",
            ])
        ].copy()

        for validation, subset in cv_folds.groupby(
            "Validation"
        ):

            subset["Experiment"] = (
                subset["FeatureConfiguration"].astype(str)
                + subset["GroupName"].fillna("").astype(str).map(
                    lambda value: f" | {value}" if value else ""
                )
            )

            labels = subset["Experiment"].unique()

            values = [
                subset.loc[
                    subset["Experiment"] == label,
                    "R2",
                ].dropna().to_numpy()
                for label in labels
            ]

            values = [
                value
                for value in values
                if len(value) > 0
            ]

            labels = [
                label
                for label in labels
                if len(
                    subset.loc[
                        subset["Experiment"] == label,
                        "R2",
                    ].dropna()
                ) > 0
            ]

            if not values:
                continue

            fig, ax = plt.subplots(
                figsize=(max(10, len(labels) * 1.5), 6)
            )

            ax.boxplot(
                values,
                tick_labels=labels,
                showmeans=True,
            )

            ax.set_title(
                f"{model_name}: Fold R² Distributions — {validation}"
            )

            ax.set_ylabel("R²")

            ax.tick_params(
                axis="x",
                rotation=30,
            )

            ax.grid(
                axis="y",
                alpha=0.25,
            )

            fig.tight_layout()

            fig.savefig(
                plot_directory
                / f"Fold_R2_{validation}.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.close(fig)

    # --------------------------------------------------------
    # Plot 3: LOCO by held-out coumarin
    # --------------------------------------------------------

    if (
        not folds_data.empty
        and "HeldOutCoumarin" in folds_data.columns
    ):

        loco = folds_data[
            folds_data["Validation"].isin([
                "LOCO",
                "GroupSpecific_LOCO",
            ])
        ]

        for validation, subset in loco.groupby(
            "Validation"
        ):

            for metric, title in metric_labels.items():

                if metric not in subset.columns:
                    continue

                fig, ax = plt.subplots(
                    figsize=(14, 7)
                )

                for configuration, config_data in subset.groupby(
                    "FeatureConfiguration"
                ):

                    for group_name, group_data in config_data.groupby(
                        "GroupName",
                        dropna=False,
                    ):

                        group_data = group_data.sort_values(
                            "HeldOutCoumarin"
                        )

                        label = configuration

                        if pd.notna(group_name):
                            label += f" | {group_name}"

                        ax.plot(
                            group_data["HeldOutCoumarin"],
                            group_data[metric],
                            marker="o",
                            label=label,
                        )

                ax.set_title(
                    f"{model_name}: {validation} {title} by Coumarin"
                )

                ax.set_xlabel("Held-out coumarin")
                ax.set_ylabel(title)

                ax.tick_params(
                    axis="x",
                    rotation=35,
                )

                ax.grid(alpha=0.25)
                ax.legend()

                fig.tight_layout()

                fig.savefig(
                    plot_directory
                    / f"{validation}_{metric}_by_Coumarin.png",
                    dpi=300,
                    bbox_inches="tight",
                )

                plt.close(fig)

    # --------------------------------------------------------
    # Plot 4: PCA component counts
    # --------------------------------------------------------

    pca_rows = []

    for (
        configuration,
        validation,
        group_name,
    ), metadata in metadata_records.items():

        pca_information = metadata.get(
            "PCAInformation",
            {},
        )

        component_count = pca_information.get(
            "PCA_Components"
        )

        if component_count is None:
            continue

        pca_rows.append({
            "FeatureConfiguration": configuration,
            "Validation": validation,
            "GroupName": group_name,
            "PCA_Components": component_count,
        })

    if pca_rows:

        pca_data = pd.DataFrame(
            pca_rows
        )

        fig, ax = plt.subplots(
            figsize=(13, 7)
        )

        labels = (
            pca_data["FeatureConfiguration"].astype(str)
            + "\n"
            + pca_data["Validation"].astype(str)
        )

        if pca_data["GroupName"].notna().any():
            labels = (
                labels
                + "\n"
                + pca_data["GroupName"].fillna("").astype(str)
            )

        ax.bar(
            labels,
            pca_data["PCA_Components"],
        )

        ax.set_title(
            f"{model_name}: Selected PCA Component Counts"
        )

        ax.set_ylabel("Number of principal components")

        ax.tick_params(
            axis="x",
            rotation=30,
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            plot_directory / "PCA_Component_Counts.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Plot 5: PCA explained variance
    # --------------------------------------------------------

    for (
        configuration,
        validation,
        group_name,
    ), metadata in metadata_records.items():

        pca_information = metadata.get(
            "PCAInformation",
            {},
        )

        ratios = pca_information.get(
            "PCA_Explained_Variance_Ratio",
            [],
        )

        if not ratios:
            continue

        ratios = np.asarray(
            ratios,
            dtype=float,
        )

        cumulative = np.cumsum(ratios)

        components = np.arange(
            1,
            len(ratios) + 1,
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.plot(
            components,
            ratios,
            marker=".",
            label="Individual explained variance",
        )

        ax.plot(
            components,
            cumulative,
            marker=".",
            label="Cumulative explained variance",
        )

        title = (
            f"{configuration} — {validation}"
        )

        if group_name:
            title += f" — {group_name}"

        ax.set_title(title)

        ax.set_xlabel("Principal component")
        ax.set_ylabel("Explained variance ratio")

        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()

        filename = (
            f"PCA_ExplainedVariance_"
            f"{configuration}_{validation}"
        )

        if group_name:
            filename += f"_{group_name}"

        fig.savefig(
            plot_directory / f"{filename}.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Save combined results
    # --------------------------------------------------------

    summary_data.to_csv(
        plot_directory / "Combined_Summary.csv",
        index=False,
    )

    if not folds_data.empty:
        folds_data.to_csv(
            plot_directory / "Combined_FoldResults.csv",
            index=False,
        )

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETED")
    print("=" * 80)

    print(
        f"Plots saved to: {plot_directory}"
    )

    print("\nCombined experiment summary:")
    print_dataframe(summary_data)


# ============================================================
# DATAFRAME DISPLAY
# ============================================================

def print_dataframe(df: pd.DataFrame):
    if df.empty:
        print("(Empty DataFrame)")
        return

    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="rounded_grid",
            showindex=False,
            floatfmt=".4f",
        )
    )